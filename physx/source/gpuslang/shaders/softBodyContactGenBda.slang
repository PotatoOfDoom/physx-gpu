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

// softBodyContactGenBda.slang
//
// P8.2 tranche a — the SOFTBODY-vs-RIGID midphase on the Vulkan engine ABI (BDA).
// Ported line-for-line from gpunarrowphase/src/CUDA/softbodyMidPhase.cu:
//
//   sb_midphaseGeneratePairsLaunch (:199, launch PxgNarrowphaseCore.cpp:4221,
//   SB_MIDPHASE_PRIMITIVES) — one warp per contact-manager pair walks the SOFTBODY's own
//   BV32 tetrahedron BVH against the RIGID's fattened world AABB (sbMidphaseCore + the
//   SoftbodyTreeTraverser callback) and appends uint4(cmIdx, tetIdx, 0, 0) to the shared
//   collision stack for every surface tetrahedron whose bounds overlap it.
//
// This is the FIRST kernel of the softbody-vs-rigid contact chain — the arc that ends with
// the softbody cube actually settling on the ground instead of falling through it
// (PXG_SCENE=softbody, which is a contact-free free fall until this chain lands).
//
// WHAT IS REUSED RATHER THAN REINVENTED:
//   - the BV32 warp-cooperative tree walk and the BV32DataPacked leaf/child decode come
//     VERBATIM from femClothSelfCollisionBda (this file's direct BDA sibling): same
//     groupshared stack, same ballot/scan lowering, same barrier placement;
//   - the SoftbodyTreeTraverser body (surface-hint gate, isValidTet, tetBoundingBox) is the
//     already-unit-tested transcription from npMidphaseDeform.slang, lifted onto Ptr<>
//     reads;
//   - PxgContactManagerInput / PxgShape are the byte-exact mirrors femClothContactGenBda
//     already carries, and the rigid AABB read (bounds[rigidCacheRef], 24 B, min @0 max @12)
//     is its precedent too.
//
// *** THE GRID.  READ THIS BEFORE CHANGING numthreads. ***
// The engine launches grid (ceil(numTests / MIDPHASE_WARPS_PER_BLOCK), 1, 1) with block
// (WARP_SIZE, MIDPHASE_WARPS_PER_BLOCK = 2, 1) and indexes cmIdx = threadIdx.y +
// blockIdx.x * blockDim.y.  The Vulkan bridge DROPS blockDim and passes only gridDim
// through (PxgVulkanCudaContext.cpp:293), so a [numthreads(32,1,1)] group is ONE warp and
// indexing by gid.x alone would silently process only HALF the contact-manager pairs.
// That exact bug shipped in femClothSelfCollisionBda and cost a long investigation; it is
// avoided here from the first line.  The block's warps are walked SERIALLY in a
// group-uniform loop, which keeps one group == one warp (so the traversal's barriers stay
// uniform — packing two warps into a group would put them in divergent control flow) while
// still covering every pair.
//
// DEVIATIONS from CUDA (recipe policy, each preserving the emitted pair SET and the atomic
// total):
//   - NO 32-lane warp intrinsics: __ballot_sync -> groupshared InterlockedOr,
//     __shfl_sync(...,0) -> groupshared broadcast, warpScanExclusive ->
//     countbits(mask & lanemask_lt).  llvmpipe's subgroup size is 8, so a 32-lane warp is
//     emulated across the group exactly as in npBv32 / femClothSelfCollisionBda;
//   - the per-warp sbMidphaseScratch (tetmeshVerts / tetmeshTetIndices /
//     tetmeshSurfaceHint / bv32PackedNodes cached by lane 0 behind a __syncwarp) is
//     dropped; every lane reads those addresses directly through Ptr<>.  Semantically
//     identical read fan-out, and it removes a barrier;
//   - mTetMeshSurfaceHint is a PxU8 array.  llvmpipe has no 8-bit storage in these shaders,
//     so the containing 32-bit word is read and the byte extracted by shift/mask — the
//     gate is `hint != 0` either way.  (Note the address is NOT assumed 4-aligned: the word
//     address is formed with `& ~uint64_t(3)`.  Writing `& ~3UL` MISCOMPILES — Slang
//     evaluates the complement in 32 bits and clears the top half of the device address.
//     That trap is documented at length in softBodySolveBda.slang and tools/lvprepro.);
//   - toleranceLength and transformCache are carried in the push blob for ABI parity but
//     NEVER read: sbMidphaseCore takes both and dereferences neither (softbodyMidPhase.cu
//     :119-135 uses only cmInputs, bounds, contactDistance, gpuShapes and softbodies).
//
// The tetmesh blob hanging off PxgSoftBody::mTetMeshData opens with a uint4 header
// (nbVerts, nbTets, maxDepth, nbBv32TreeNodes), so the BV32 node array starts at
// mTetMeshData + 16 — the same +16 skip softBodyRefitBda and femClothSelfCollisionBda use.
//
// This kernel emits pairs; it computes no contact geometry.  Every decision is a min/max
// or a comparison of loaded values, EXCEPT isValidTet's signed volume (a dot of a cross).
// It is NoContraction-stamped (PXG_NOCONTRACT_SHADERS) so that one arithmetic gate stays
// bit-exact against the -ffp-contract=off CPU reference; there is no transcendental
// anywhere, so this tranche is FULLY bit-exact with no tolerance seam.

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu (mTetMeshData@0, mTetIndices@16, ...)

// PxgNpKernelIndices.h:32 — the engine's warps-per-block for every midphase launch.
static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
static const uint WARP = 32u;

// PxGeometryType::eTETRAHEDRONMESH (PxGeometry.h:57) — 8th enumerator, zero-based.
static const uint GEOM_TETRAHEDRONMESH = 7u;

// Gu::BV32DataPacked: 1168 B — mMin[32] float4 @0, mMax[32] float4 @512, mData[32] uint
// @1024, mNbNodes @1152.
static const uint64_t BV32_STRIDE = 1168UL;
static const uint64_t BV32_MMAX_OFFSET = 512UL;
static const uint64_t BV32_MDATA_OFFSET = 1024UL;
static const uint64_t BV32_MNBNODES_OFFSET = 1152UL;
static const uint GU_BV4_CHILD_OFFSET_SHIFT_COUNT = 11u;

// femMidphaseScratch.sBv32Nodes[192] — a 6-deep BV32 tree's worst-case pending set.
static const uint STACK_SIZE = 192u;

// ---------------------------------------------------------------------------
// Byte-exact narrowphase ABI mirrors (identical to femClothContactGenBda's).
// ---------------------------------------------------------------------------
struct PxgContactManagerInput   // PxgContactManager.h, 16 B
{
    uint shapeRef0;          // @0
    uint shapeRef1;          // @4
    uint transformCacheRef0; // @8
    uint transformCacheRef1; // @12
}

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
// sb_midphaseGeneratePairsLaunch push blob (softbodyMidPhase.cu:199).
// CUDA arg pack (PxgNarrowphaseCore.cpp:4223): { PxU32 numTests, PxReal toleranceLength,
// const PxgContactManagerInput* cmInputs, const PxsCachedTransform* transformCache,
// const PxBounds3* bounds, const PxReal* contactDistance, const PxgShape* gpuShapes,
// const PxgSoftBody* softbodies, PxU32 stackSizeBytes, PxU8* stackPtr,
// PxU32* midphasePairsNum } -> natural alignment: 4@0, 4@4, then six pointers @8..@48,
// PxU32 @56, then a pointer realigned to @64 and @72 -> 80-byte blob.
// (ABI row {4,4,8,8,8,8,8,8,4,8,8}.)
// ---------------------------------------------------------------------------
struct SbMidphaseParams
{
    uint numTests;              // @0
    float toleranceLength;      // @4   (ABI parity only — never read)
    uint64_t cmInputs;          // @8
    uint64_t transformCache;    // @16  (ABI parity only — never read)
    uint64_t bounds;            // @24
    uint64_t contactDistance;   // @32
    uint64_t gpuShapes;         // @40
    uint64_t softbodies;        // @48
    uint stackSizeBytes;        // @56
    uint _pad60;                // @60
    uint64_t stackPtr;          // @64  output
    uint64_t midphasePairsNum;  // @72  output -> 80
}
[[vk::push_constant]]
ConstantBuffer<SbMidphaseParams> gMid;

// ---------------------------------------------------------------------------
// BV32DataPacked field readers — explicit byte offsets through Ptr<>, no whole-node copy.
// treeAddr = mTetMeshData + sizeof(uint4).
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

// isValidTet (cudaNpCommon.h): strictly positive signed volume.
bool isValidTet(float3 v0, float3 v1, float3 v2, float3 v3)
{
    return dot(v1 - v0, cross(v2 - v0, v3 - v0)) > 1e-9f;
}

// PxBounds3::intersects — the negated six-comparison form, exactly as PxBounds3.h writes it.
bool boundsIntersect(float3 aMin, float3 aMax, float3 bMin, float3 bMax)
{
    return !(bMin.x > aMax.x || aMin.x > bMax.x ||
             bMin.y > aMax.y || aMin.y > bMax.y ||
             bMin.z > aMax.z || aMin.z > bMax.z);
}

// mTetMeshSurfaceHint is a PxU8 array; read the containing 32-bit word and extract the
// byte.  See the header on `& ~uint64_t(3)` — `& ~3UL` silently truncates the address.
uint surfaceHint(uint64_t hintBase, uint primIdx)
{
    const uint64_t addr = hintBase + uint64_t(primIdx);
    const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
    const uint shift = uint(addr & 3UL) * 8u;
    return (word >> shift) & 0xffu;
}

// LoadShapePairWarp<PxGeometryType::eTETRAHEDRONMESH> (dataReadWriteHelper.cuh): shape0 is
// whichever of the pair IS the tetrahedron mesh; the other is the rigid.  Getting this
// backwards silently reads the rigid's id as a softbody index, so it is written out
// explicitly rather than assumed.
void loadSoftbodyRigidPair(uint64_t cmInputsAddr, uint64_t shapesAddr, uint cmIdx,
                           out PxgShape sbShape, out uint sbCacheRef,
                           out PxgShape rigidShape, out uint rigidCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(cmInputsAddr)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(shapesAddr);
    uint sRefSb = cin.shapeRef0, cRefSb = cin.transformCacheRef0;
    uint sRefRigid = cin.shapeRef1, cRefRigid = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != GEOM_TETRAHEDRONMESH)
    {
        sRefSb = cin.shapeRef1; cRefSb = cin.transformCacheRef1;
        sRefRigid = cin.shapeRef0; cRefRigid = cin.transformCacheRef0;
    }
    sbShape = shapes[sRefSb]; sbCacheRef = cRefSb;
    rigidShape = shapes[sRefRigid]; rigidCacheRef = cRefRigid;
}

// ---------------------------------------------------------------------------
// Warp emulation scratch: one group == one warp, so a single slot each.
// ---------------------------------------------------------------------------
groupshared int sStack[STACK_SIZE];   // bv32 to-be-expanded node stack
groupshared uint sBallot;             // __ballot_sync mask
groupshared uint sScanStart;          // atomicAdd result broadcast

uint ballotGroup(bool pred, uint lane)
{
    if (lane == 0u) sBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sBallot;
}

// ===========================================================================
// sb_midphaseGeneratePairsLaunch — one 32-thread group per warp, walked serially over
// the block's MIDPHASE_WARPS_PER_BLOCK warps.
// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void sb_midphaseGeneratePairsLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    // See the header: the engine packs MIDPHASE_WARPS_PER_BLOCK warps per block and the
    // bridge forwards only gridDim, so the block's warps are walked serially here.  `sub`
    // and the break are group-uniform, so every barrier below is reached uniformly.
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    GroupMemoryBarrierWithGroupSync();   // groupshared scratch is reused per sub-warp
    const uint cmIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;
    if (cmIdx >= gMid.numTests)
        break;

    PxgShape sbShape, rigidShape;
    uint sbCacheRef, rigidCacheRef;
    loadSoftbodyRigidPair(gMid.cmInputs, gMid.gpuShapes, cmIdx,
                          sbShape, sbCacheRef, rigidShape, rigidCacheRef);

    const uint softbodyId = sbShape.particleOrSoftbodyId;
    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gMid.softbodies);

    const uint64_t tetMeshData = softbodies[softbodyId].mTetMeshData;
    const uint64_t treeAddr = tetMeshData + 16UL;   // + sizeof(uint4) header

    Ptr<float4> tetmeshVerts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);
    Ptr<uint4> tetmeshTetIndices = Ptr<uint4>(softbodies[softbodyId].mTetIndices);
    const uint64_t hintBase = softbodies[softbodyId].mTetMeshSurfaceHint;

    // The RIGID's world AABB, fattened by the pair's total contact distance
    // (softbodyMidPhase.cu:186-190).  PxBounds3 is 24 B: min @0, max @12.
    Ptr<float> bounds = Ptr<float>(gMid.bounds);
    const uint rb = rigidCacheRef * 6u;
    Ptr<float> contactDistance = Ptr<float>(gMid.contactDistance);
    const float contactDist = contactDistance[sbCacheRef] + contactDistance[rigidCacheRef];
    const float3 boxMin = float3(bounds[rb + 0u], bounds[rb + 1u], bounds[rb + 2u]) - contactDist;
    const float3 boxMax = float3(bounds[rb + 3u], bounds[rb + 4u], bounds[rb + 5u]) + contactDist;

    const uint stackSize = gMid.stackSizeBytes / 16u;   // stackSizeBytes / sizeof(uint4)

    uint currentNodeIndex = 0u;   // root
    bool hasNode = true;
    uint nbStack = 0u;

    while (hasNode)
    {
        const uint nodeIdx = currentNodeIndex >> 6u;

        if ((currentNodeIndex & 1u) != 0u)
        {
            // Leaf: the primitive range lives in the PARENT node's mData slot.
            const uint leafOffset = (currentNodeIndex >> 1u) & 31u;
            const uint data = bvData(treeAddr, nodeIdx, leafOffset);
            const uint nbPrimitives = (data >> 1u) & 63u;
            const uint primStartIndex = data >> 7u;
            uint primitiveIndex = 0xffffffffu;
            if (lane < nbPrimitives)
                primitiveIndex = lane + primStartIndex;

            // SoftbodyTreeTraverser::intersectPrimitiveFullWarp (softbodyMidPhase.cu:82).
            bool intersect = false;
            if (primitiveIndex != 0xffffffffu)
            {
                // Only SURFACE tetrahedra can touch a rigid; the interior is skipped.
                if (surfaceHint(hintBase, primitiveIndex) != 0u)
                {
                    const uint4 tetIdx = tetmeshTetIndices[primitiveIndex];
                    const float3 w0 = tetmeshVerts[tetIdx.x].xyz;
                    const float3 w1 = tetmeshVerts[tetIdx.y].xyz;
                    const float3 w2 = tetmeshVerts[tetIdx.z].xyz;
                    const float3 w3 = tetmeshVerts[tetIdx.w].xyz;

                    if (isValidTet(w0, w1, w2, w3))
                    {
                        // tetBoundingBox (deformableCollision.cuh:64) — pairwise min/max,
                        // transcribed in the same association order as the .cu.
                        const float3 tMin = min(min(w0, w1), min(w2, w3));
                        const float3 tMax = max(max(w0, w1), max(w2, w3));
                        intersect = boundsIntersect(tMin, tMax, boxMin, boxMax);
                    }
                }
            }

            // pushOntoStackFullWarp (bv32Traversal.cuh:197): reserve a run with one
            // atomicAdd, scatter by the exclusive prefix, DROP anything past stackSize.
            const uint mask = ballotGroup(intersect, lane);
            const uint offset = countbits(mask & ((1u << lane) - 1u));
            const uint validCount = countbits(mask);
            if (lane == 0u)
            {
                uint start = 0u;
                if (validCount > 0u)
                    InterlockedAdd(Ptr<uint>(gMid.midphasePairsNum)[0], validCount, start);
                sScanStart = start;
            }
            GroupMemoryBarrierWithGroupSync();
            const uint stackIndex = sScanStart + offset;
            if (intersect && stackIndex < stackSize)
                Ptr<uint4>(gMid.stackPtr)[stackIndex] = uint4(cmIdx, primitiveIndex, 0u, 0u);
        }
        else
        {
            // Internal node: test the rigid box against each child's bounds.
            const uint nbChildren = bvNbNodes(treeAddr, nodeIdx);
            float3 mn = float3(0, 0, 0), mx = float3(0, 0, 0);
            const bool hasBox = lane < nbChildren;
            if (hasBox)
            {
                mn = bvMin(treeAddr, nodeIdx, lane);
                mx = bvMax(treeAddr, nodeIdx, lane);
            }
            // SoftbodyTreeTraverser::intersectBoxFullWarp: mBox.intersects(childBounds).
            bool intersect = false;
            if (hasBox)
                intersect = boundsIntersect(boxMin, boxMax, mn, mx);

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
// P8.2 tranche b — sb_primitiveContactGenLaunch (softbodyPrimitives.cu:573, launch
// PxgNarrowphaseCore.cpp:4287, SB_PRIMITIVES_CG).  Turns the midphase's (cmIdx, tetIdx)
// pairs into real contact points.
//
// *** THIS PORT IMPLEMENTS THE PLANE PATH ONLY.  READ THIS BEFORE USING IT. ***
// The CUDA kernel branches on the rigid shape's geometry type.  ePLANE goes through
// tetPlaneCollide (softbodyPrimitives.cu:247), which is four lanes testing the tet's four
// corners against the plane -- small, self-contained, no GJK.  Every other type (eSPHERE,
// eCAPSULE, eBOX, eCONVEXMESH) goes through GJK + EPA over squawk::EpaScratch and
// TetCollideScratch, which is a much larger body of machinery and its own tranche.
//
// Here the non-plane branch emits NOTHING and returns.  That is a deliberate, documented
// partial implementation, not an oversight:
//   - it is NOT a regression.  Until this tranche the whole kernel was unmapped, so every
//     rigid type produced zero softbody contacts.  Plane now works; the rest is unchanged;
//   - it IS silent from the engine's point of view -- a box ground will simply generate no
//     contacts and a softbody will fall through it, exactly as it does today.  Anything
//     relying on softbody-vs-box contact must wait for the GJK tranche.
// PXG_SCENE=softbody drops onto a PxPlaneGeometry precisely so this path is the one under
// test.
//
// DEVIATIONS from CUDA:
//   - the fixed grid is reproduced as a constant.  The engine hard-codes numBlocks = 512
//     (PxgNarrowphaseCore.cpp:4300) and blockDim (WARP_SIZE, MIDPHASE_WARPS_PER_BLOCK, 1);
//     the bridge forwards only gridDim, so the grid-stride `i += gridDim.x * blockDim.y`
//     becomes `i += SB_CG_GRID * MIDPHASE_WARPS_PER_BLOCK`.  Same precedent as
//     femClothRigidContactBda's RC_GRID / RC_BLOCK;
//   - as in the midphase above, the block's two warps are walked SERIALLY in a
//     group-uniform loop rather than packed into one group;
//   - the per-warp sbTetDataScratch pointer cache and the 12-lane cooperative load of the
//     tet's four vertices into ss_scratch.vB are dropped: lanes 0..3 each read their own
//     corner directly through Ptr<>.  Same values, no shared scratch, no barrier;
//   - PxgFEMContactWriter is passed BY VALUE as the last CUDA argument, so it is INLINED
//     into the push blob (fields at @112..@184), exactly as femClothContactGenBda does.
//     That makes the blob 192 bytes.  llvmpipe offers maxPushConstantsSize = 256 so it
//     fits, but the Vulkan MINIMUM is 128 -- this is the first kernel in the port that
//     would not load on a device at the guaranteed floor.  Splitting the writer into a
//     descriptor set would be the fix if that ever matters.
//
// NoContraction-stamped: transformInv's rotate-by-conjugate and the plane-space dot are
// mul-feeding-add chains.  No transcendental -- fully bit-exact, no tolerance seam.
// ===========================================================================

// PxgNarrowphaseCore.cpp:4300 — the engine hard-codes this grid for SB_PRIMITIVES_CG.
static const uint SB_CG_GRID = 512u;

// PxGeometryType::ePLANE (PxGeometry.h:51).
static const uint GEOM_PLANE = 1u;

// PxgRigidFilterPair (PxgSimulationCoreDesc.h, 32 B): index0 u64 @0, index1 u64 @8,
// index2 u32 @16, tail padding.
struct PairR { uint2 index0; uint2 index1; uint index2; uint pad0, pad1, pad2; }

struct SbContactGenParams
{
    float tolerenceLength;          // @0    (ABI parity only — the plane path never reads it)
    uint _pad4;                     // @4
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16
    uint64_t contactDistance;       // @24
    uint64_t restDistance;          // @32
    uint64_t gpuShapes;             // @40
    uint64_t softbodies;            // @48
    uint64_t shapeToRigidRemapTable;// @56
    uint64_t stackPtr;              // @64
    uint64_t midphasePairsNum;      // @72
    uint stackSizeBytes;            // @80
    uint _pad84;                    // @84
    uint64_t softBodyRigidPairs;    // @88
    uint numSoftBodyRigidPairs;     // @96
    uint _pad100;                   // @100
    uint64_t stackSizeNeededOnDevice; // @104
    // --- PxgFEMContactWriter, by value (PxgFEMCore.h:361) ---
    uint64_t w_outPoint;            // @112 float4*
    uint64_t w_outNormalPen;        // @120 float4*
    uint64_t w_outBarycentric;      // @128 float4* (unused on the no-barycentric path)
    uint64_t w_outContactInfo;      // @136 PxgFemOtherContactInfo* (16 B stride)
    uint64_t w_totalContactCount;   // @144 PxU32*
    uint64_t w_contactByX;          // @152 PxU64*
    uint64_t w_tempContactByX;      // @160 PxU32*
    uint64_t w_contactIndexSortedByX; // @168 PxU32*
    uint64_t w_contactSortedByX;    // @176 PxU32* (unused here)
    uint w_maxNumContacts;          // @184 PxU32
    uint _pad188;                   // @188 -> 192
}
[[vk::push_constant]]
ConstantBuffer<SbContactGenParams> gCG;

groupshared uint sCgBallot;
groupshared uint sCgStart;

uint cgBallot(bool pred, uint lane)
{
    if (lane == 0u) sCgBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sCgBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sCgBallot;
}

// PxgRigidFilterPair binary-search find with the ignoreIndex2 early accept
// (deformableElementFilter.cuh:67/:89) — the same transcription femClothContactGenBda uses.
bool u64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
bool u64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

bool findRigidPair(uint64_t pairsAddr, uint nbPairs, uint2 q0, uint2 q1)
{
    Ptr<PairR> pairs = Ptr<PairR>(pairsAddr);
    uint l = 0u, r = nbPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2u;
        const PairR p = pairs[mid];
        int cmp = 0;
        if (u64Lt(q0, p.index0)) cmp = -1;
        else if (u64Lt(p.index0, q0)) cmp = 1;
        else if (u64Lt(q1, p.index1)) cmp = -1;
        else if (u64Lt(p.index1, q1)) cmp = 1;
        else if (0u < p.index2) cmp = -1;   // q.index2 == 0
        if (cmp == 0 || (u64Eq(q0, p.index0) && u64Eq(q1, p.index1)))
            return true;
        if (cmp < 0) r = mid; else l = mid + 1u;
    }
    return false;
}

// PxgFEMContactWriter::writeRigidVsDeformableContactNoBarycentric (PxgFEMCore.h).
// setRigidMaterialIndex(mat) followed by markInCollision(false) collapses to a single
// store of (mat & 0x7fffffff) — the same fold femClothContactGenBda documents.
void writeRigidVsDeformableNoBary(uint index, float4 contact, float4 normalPen,
                                  uint2 pairInd0, uint pairInd1, uint2 rigidId, uint rigidMatId)
{
    if (index >= gCG.w_maxNumContacts)
        return;
    Ptr<uint2>(gCG.w_contactByX)[index] = rigidId;
    Ptr<uint>(gCG.w_tempContactByX)[index] = rigidId.x;   // PxU32(rigidId & 0xffffffff)
    Ptr<uint>(gCG.w_contactIndexSortedByX)[index] = index;
    Ptr<float4>(gCG.w_outPoint)[index] = contact;
    Ptr<float4>(gCG.w_outNormalPen)[index] = normalPen;
    // PxgFemOtherContactInfo (16 B): pairInd0 u64 @0, pairInd1 u32 @8, rigidMatInd @12.
    const uint64_t ci = gCG.w_outContactInfo + uint64_t(index) * 16UL;
    Ptr<uint2>(ci)[0] = pairInd0;
    Ptr<uint>(ci + 8UL)[0] = pairInd1;
    Ptr<uint>(ci + 12UL)[0] = rigidMatId & 0x7fffffffu;
}

// PxTransform::transformInv(p) = q.rotateInv(p - t); rotateInv is the conjugate rotate.
float3 quatRotateInv(float4 q, float3 v)
{
    const float w = q.w;
    const float3 u = float3(q.x, q.y, q.z);
    const float vx = w * w - dot(u, u);
    return v * vx - cross(u, v) * (w * 2.0f) + u * (dot(u, v) * 2.0f);
}
// PxQuat::getBasisVector0() — the rotated x axis.
float3 quatBasisVector0(float4 q)
{
    const float x2 = q.x * 2.0f, w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2,
                  (q.z * w2) + q.y * x2,
                  (-q.y * w2) + q.z * x2);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void sb_primitiveContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    const uint stackSize = gCG.stackSizeBytes / 16u;
    const uint numFound = Ptr<uint>(gCG.midphasePairsNum)[0];
    const uint numPairs = min(numFound, stackSize);

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gCG.softbodies);
    Ptr<float> contactDistance = Ptr<float>(gCG.contactDistance);
    Ptr<float> restDistance = Ptr<float>(gCG.restDistance);

    // The block's two warps, walked serially (see the header).
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
        const uint globalWarpIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;
        const uint stride = SB_CG_GRID * MIDPHASE_WARPS_PER_BLOCK;

        for (uint i = globalWarpIdx; i < numPairs; i += stride)
        {
            GroupMemoryBarrierWithGroupSync();   // groupshared ballot is reused per pair

            const uint4 curPair = Ptr<uint4>(gCG.stackPtr)[i];
            const uint cmIdx = curPair.x;
            const uint tetrahedronIdx = curPair.y;

            PxgShape sbShape, rigidShape;
            uint sbCacheRef, rigidCacheRef;
            loadSoftbodyRigidPair(gCG.cmInputs, gCG.gpuShapes, cmIdx,
                                  sbShape, sbCacheRef, rigidShape, rigidCacheRef);

            const float cDistance = contactDistance[sbCacheRef] + contactDistance[rigidCacheRef];
            const float restDist = restDistance[cmIdx];

            // PxNodeIndex is a PxU64; the remap table is indexed by the rigid's cache ref.
            const uint2 rigidId = Ptr<uint2>(gCG.shapeToRigidRemapTable)[rigidCacheRef];
            const uint softbodyId = sbShape.particleOrSoftbodyId;
            // PxEncodeSoftBodyIndex (PxgSimulationCoreDesc.h:110): (id << 20) | tetId.
            const uint softBodyMask = (softbodyId << 20u) | tetrahedronIdx;

            // An ATTACHED tetrahedron must not also generate a contact against the same
            // rigid — the attachment constraint already binds them.
            //
            // NOTE THE `numSoftBodyRigidPairs > 0` GUARD, and do not remove it as redundant.
            // findRigidPair's binary search cannot execute its body when nbPairs == 0
            // (l == r == 0), so by C semantics the pointer is never dereferenced -- but the
            // load inside the loop gets SPECULATED, and with a null filter-pair address that
            // faults the device.  Bisected: a probe placed just before this call survives,
            // one placed just after it SIGSEGVs.  Same family as the `~3UL` trap documented
            // in softBodySolveBda -- a guard the source language says is unnecessary and the
            // backend says is not.
            if (gCG.numSoftBodyRigidPairs > 0u &&
                findRigidPair(gCG.softBodyRigidPairs, gCG.numSoftBodyRigidPairs,
                              rigidId, uint2(softBodyMask, 0u)))
                continue;

            // *** PLANE ONLY — see the header.  Everything else is the GJK tranche. ***
            if (rigidShape.type != GEOM_PLANE)
                continue;

            // tetPlaneCollide (softbodyPrimitives.cu:247): lanes 0..3 take the tet's four
            // corners.  CUDA stages them through ss_scratch.vB via a 12-lane cooperative
            // load; each lane reads its own corner directly here.
            const uint4 tetIdx = Ptr<uint4>(softbodies[softbodyId].mTetIndices)[tetrahedronIdx];
            Ptr<float4> verts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);

            // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
            const uint64_t xf = gCG.transformCache + uint64_t(rigidCacheRef) * 32UL;
            const float4 planeQ = Ptr<float4>(xf)[0];
            const float3 planeP = float3(Ptr<float>(xf + 16UL)[0],
                                         Ptr<float>(xf + 20UL)[0],
                                         Ptr<float>(xf + 24UL)[0]);

            bool intersect = false;
            float3 worldPoint = float3(0, 0, 0);
            float3 worldNormal = float3(0, 0, 0);
            float pen = 0.0f;
            if (lane < 4u)
            {
                uint corner = tetIdx.x;
                if (lane == 1u) corner = tetIdx.y;
                else if (lane == 2u) corner = tetIdx.z;
                else if (lane == 3u) corner = tetIdx.w;

                const float3 wp = verts[corner].xyz;
                // A PxPlane's surface is x == 0 in plane space, so the signed distance is
                // the transformed point's x component.
                const float3 pInPlaneSpace = quatRotateInv(planeQ, wp - planeP);
                if (cDistance >= pInPlaneSpace.x)
                {
                    intersect = true;
                    worldPoint = wp;
                    worldNormal = quatBasisVector0(planeQ);
                    pen = pInPlaneSpace.x;
                }
            }

            // globalScanExclusiveSingleWarp(intersect, writer.totalContactCount).
            const uint mask = cgBallot(intersect, lane);
            const uint offset = countbits(mask & ((1u << lane) - 1u));
            const uint validCount = countbits(mask);
            if (lane == 0u)
            {
                uint start = 0u;
                if (validCount > 0u)
                    InterlockedAdd(Ptr<uint>(gCG.w_totalContactCount)[0], validCount, start);
                sCgStart = start;
            }
            GroupMemoryBarrierWithGroupSync();
            const uint index = sCgStart + offset;

            if (intersect && index < gCG.w_maxNumContacts)
            {
                writeRigidVsDeformableNoBary(index,
                                             float4(worldPoint.x, worldPoint.y, worldPoint.z, restDist),
                                             float4(worldNormal.x, worldNormal.y, worldNormal.z, pen),
                                             rigidId, softBodyMask, rigidId, rigidShape.materialIndex);
            }
        }
    }

    // atomicMax(stackSizeNeededOnDevice, numMidphasePairsFound * sizeof(uint4)) — how the
    // engine learns to grow the collision stack for the next frame.
    if (gid.x == 0u && lane == 0u)
        InterlockedMax(Ptr<uint>(gCG.stackSizeNeededOnDevice)[0], numFound * 16u);
}
