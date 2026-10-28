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
//   - the BV32DataPacked leaf/child decode is pxgSlangBv32's; the tree walk is the shared
//     pxgSlangBv32Batched one (also cloth_midphaseGeneratePairsLaunch's), which replaced a
//     one-warp transcription of femClothSelfCollisionBda's walk;
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
// group-uniform loop (so the traversal's barriers stay uniform) while still covering every
// pair.  Since the batched walk (pxgSlangBv32Batched) the group is 1024 threads, and EACH
// contact manager is walked by the whole group, not by one 32-thread warp.
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
import pxgSlangDefFilter;   // 5.12: findRigidVolumeFilter
import pxgSlangDeformableRef;     // PxgEncodeDeformableTetrahedron
import pxgSlangDeformableWriters; // PxgRigidDbContactWriterGpu
import pxgSlangGjk;          // GjkOut, GJK_* results, GJK_MAX_CLOUD
import pxgSlangTetCollide;   // computeInnerSphere
import pxgSlangGjkWarp;      // gwTetPrimitivesCollide2: GJK/EPA on the whole warp
import pxgSlangBv32Batched;   // the 1024-thread midphase walk (sb_midphaseGeneratePairsLaunch)
import pxgSlangBv32;   // P9.17b: the BV32 reader AND its index rules, replacing a
                      // private copy of both that was verbatim identical to it.

// PxgNpKernelIndices.h:32 — the engine's warps-per-block for every midphase launch.
static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
static const uint WARP = 32u;

// PxGeometryType::eTETRAHEDRONMESH (PxGeometry.h:57) — 8th enumerator, zero-based.
static const uint GEOM_TETRAHEDRONMESH = 7u;

// @1024, mNbNodes @1152.


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
    uint numWorkItems;          // @0  (CUDA softbodyMidPhase.cu:200 `numWorkItems`)
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
// SoftbodyTreeTraverser::intersectPrimitiveFullWarp (softbodyMidPhase.cu:82) for one
// tetrahedron: only SURFACE tets, only valid ones, bound from the four live corners.
// ---------------------------------------------------------------------------
struct SbTetLeaf : IBv32BatchedLeaf
{
    Ptr<float4> tetmeshVerts;
    Ptr<uint4> tetmeshTetIndices;
    uint64_t hintBase;
    float3 boxMin;
    float3 boxMax;

    bool overlaps(uint primitiveIndex)
    {
        bool intersect = false;
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
        return intersect;
    }
}

// ===========================================================================
// sb_midphaseGeneratePairsLaunch — the block's MIDPHASE_WARPS_PER_BLOCK contact managers
// walked one after the other, each by the whole 1024-thread group (pxgSlangBv32Batched).
//
// The CUDA's one warp pops the tree one node at a time: one serial dependent iteration
// per LEAF (hint -> tet indices -> corners -> atomicAdd).  For a softbody on a ground
// PLANE every surface leaf is reached, so that was the whole surface walked serially on
// one compute unit, per softbody, per step.  The batched walk expands up to 32 nodes per
// round and evaluates up to 1024 leaves per flush with one atomicAdd, and emits the same
// pairs into the same slots for a contact manager, in the same order (the module header
// has the argument; softBodySimMidphaseOrder pins it against a host model of the
// one-warp walk).  The CUDA's 192-entry stack is the module's BV32B_RESERVE.
// ===========================================================================
[shader("compute")]
[numthreads(1024, 1, 1)]   // BV32B_THREADS
void sb_midphaseGeneratePairsLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;

    // See the header: the engine packs MIDPHASE_WARPS_PER_BLOCK warps per block and the
    // bridge forwards only gridDim, so the block's managers are walked serially here.
    // `sub` and the break are group-uniform, so every barrier in the walk is reached
    // uniformly.
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    const uint cmIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;
    if (cmIdx >= gMid.numWorkItems)
        break;
    // llvmpipe runs the rest of this iteration with an EMPTY mask after the uniform
    // break and still issues its loads (P9.22): reading cmInputs[numWorkItems] and
    // then a shape at whatever index that garbage names faulted in the
    // three-manager fixture.  The loads use a clamped index; every store and
    // atomic is masked, so the clamp is unobservable.
    const uint cmLoad = min(cmIdx, gMid.numWorkItems - 1u);

    PxgShape sbShape, rigidShape;
    uint sbCacheRef, rigidCacheRef;
    loadSoftbodyRigidPair(gMid.cmInputs, gMid.gpuShapes, cmLoad,
                          sbShape, sbCacheRef, rigidShape, rigidCacheRef);

    const uint softbodyId = sbShape.particleOrSoftbodyId;
    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gMid.softbodies);

    const uint64_t tetMeshData = softbodies[softbodyId].mTetMeshData;
    const uint64_t treeAddr = tetMeshData + 16UL;   // + sizeof(uint4) header

    // The RIGID's world AABB, fattened by the pair's total contact distance
    // (softbodyMidPhase.cu:186-190).  PxBounds3 is 24 B: min @0, max @12.
    Ptr<float> bounds = Ptr<float>(gMid.bounds);
    const uint rb = rigidCacheRef * 6u;
    Ptr<float> contactDistance = Ptr<float>(gMid.contactDistance);
    const float contactDist = contactDistance[sbCacheRef] + contactDistance[rigidCacheRef];
    const float3 boxMin = float3(bounds[rb + 0u], bounds[rb + 1u], bounds[rb + 2u]) - contactDist;
    const float3 boxMax = float3(bounds[rb + 3u], bounds[rb + 4u], bounds[rb + 5u]) + contactDist;

    SbTetLeaf leaf;
    leaf.tetmeshVerts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);
    leaf.tetmeshTetIndices = Ptr<uint4>(softbodies[softbodyId].mTetIndices);
    leaf.hintBase = softbodies[softbodyId].mTetMeshSurfaceHint;
    leaf.boxMin = boxMin;
    leaf.boxMax = boxMax;

    // stackSizeBytes / sizeof(uint4); the counter overshoot contract lives in the flush.
    bv32BatchedMidphase(leaf, tid, treeAddr, boxMin, boxMax, cmIdx,
                        gMid.stackPtr, gMid.midphasePairsNum, gMid.stackSizeBytes / 16u);
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
//   - ONE WORKGROUP PER CUDA WARP.  The engine launches 512 blocks of (WARP_SIZE,
//     MIDPHASE_WARPS_PER_BLOCK = 2, 1); the bridge forwards only gridDim, so the block's
//     two warps used to share one 32-thread group and run one after the other (half the
//     CUDA's warps in flight, every pair behind another).  The port's launch site
//     (PxgNarrowphaseCore.cpp, under PXG_NO_CUDA_TOOLKIT) now launches 512 * 2 = 1024
//     one-warp groups: globalWarpIdx = gid.x and the CUDA's grid-stride
//     `i += gridDim.x * blockDim.y` is `i += WorkgroupCount().x`, so every warp walks
//     exactly the pairs its CUDA warp walks (and at any grid the loop still covers them);
//   - the per-warp sbTetDataScratch pointer cache and the 12-lane cooperative load of the
//     tet's four vertices into ss_scratch.vB are dropped: lanes 0..3 each read their own
//     corner directly through Ptr<>.  Same values, no shared scratch, no barrier;
//   - PxgRigidDbContactWriter (5.12; PxgFEMContactWriter in 5.6.1) is passed BY VALUE as
//     the last CUDA argument, so it is INLINED into the push blob (fields at @96..@168),
//     exactly as femClothContactGenBda does.  That makes the blob 176 bytes.  llvmpipe
//     offers maxPushConstantsSize = 256 so it fits, but the Vulkan MINIMUM is 128 -- a
//     device at the guaranteed floor would not load it.  Splitting the writer into a
//     descriptor set would be the fix if that ever matters;
//   - PhysX 5.12: the rigid filter is per softbody (mRigidVolumeFilterPairs, searched by
//     findRigidVolumeFilter with the rigid's cache ref) instead of a kernel argument, and
//     the contact refs are PxgDeformableRef (PxgEncodeDeformableTetrahedron).
//
// NoContraction-stamped: transformInv's rotate-by-conjugate and the plane-space dot are
// mul-feeding-add chains.  No transcendental -- fully bit-exact, no tolerance seam.
// ===========================================================================

// PxGeometryType (PxGeometry.h:48-64): eSPHERE 0, ePLANE 1, eCAPSULE 2, eBOX 3,
// eCONVEXCORE 4, eCONVEXMESH 5, ePARTICLESYSTEM 6, eTETRAHEDRONMESH 7,
// eTRIANGLEMESH 8, eHEIGHTFIELD 9.
//
// P9.57: eCONVEXMESH said 4, which is eCONVEXCORE -- the enumeration written out
// above it omitted eCONVEXCORE and so shifted everything above eBOX down by one.
// The consequence was live: the GJK dispatch at the bottom of this file admits
// SPHERE, CAPSULE, BOX and CONVEXMESH and `continue`s on everything that is not a
// PLANE, so a rigid CONVEX MESH (type 5) failed the test and its softbody contact
// was dropped IN SILENCE, while an eCONVEXCORE shape was routed into the hull
// path and would have read a blob it does not have.
static const uint GEOM_PLANE = 1u;
static const uint GEOM_SPHERE = 0u;
static const uint GEOM_CAPSULE = 2u;
static const uint GEOM_BOX = 3u;
static const uint GEOM_CONVEXMESH = 5u;

// sb_primitiveContactGenLaunch (softbodyPrimitives.cu, 5.12): { PxReal tolerenceLength,
// cmInputs*, transformCache*, contactDistance*, restDistance*, gpuShapes*, softbodies*,
// shapeToRigidRemapTable*, stackPtr*, midphasePairsNum*, PxU32 stackSizeBytes,
// stackSizeNeededOnDevice*, PxgRigidDbContactWriter writer } -> 176-byte blob.
// (ABI row {4,8,8,8,8,8,8,8,8,8,4,8,80}.)
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
    uint64_t stackSizeNeededOnDevice; // @88
    // --- PxgRigidDbContactWriter, by value (PxgDeformableContactWriters.h, 5.12) ---
    uint64_t w_outPoint;                  // @96  float4*
    uint64_t w_outNormalPen;              // @104 float4*
    uint64_t w_outBarycentric;            // @112 float4* (unused: writeVolumeContact)
    uint64_t w_outContactInfo;            // @120 PxgDeformablePairInfo*
    uint64_t w_outRigidMaterialIndex;     // @128 PxU16*
    uint64_t w_totalContactCount;         // @136 PxU32*
    uint64_t w_contactByRigid;            // @144 PxU64*
    uint64_t w_tempContactByRigid;        // @152 PxU32*
    uint64_t w_contactIndexSortedByRigid; // @160 PxU32*
    uint w_maxNumContacts;                // @168 PxU32
    uint _pad172;                         // @172 -> 176
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


PxgRigidDbContactWriterGpu cgWriter()
{
    PxgRigidDbContactWriterGpu w;
    w.outPoint = gCG.w_outPoint;                 w.outNormalPen = gCG.w_outNormalPen;
    w.outBarycentric = gCG.w_outBarycentric;     w.outContactInfo = gCG.w_outContactInfo;
    w.outRigidMaterialIndex = gCG.w_outRigidMaterialIndex;
    w.totalContactCount = gCG.w_totalContactCount;
    w.contactByRigid = gCG.w_contactByRigid;     w.tempContactByRigid = gCG.w_tempContactByRigid;
    w.contactIndexSortedByRigid = gCG.w_contactIndexSortedByRigid;
    w.maxNumContacts = gCG.w_maxNumContacts;
    return w;
}

// PxTransform::transformInv(p) = q.rotateInv(p - t); rotateInv is the conjugate rotate.
// PxQuat::rotate (PxQuat.h:286).
float3 cgQuatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// P9.37 REMOVED a second, algebraically-equivalent rotateInv from this file.  It was
//
//     const float w = q.w; const float3 u = q.xyz;
//     const float vx = w * w - dot(u, u);
//     return v * vx - cross(u, v) * (w * 2.0f) + u * (dot(u, v) * 2.0f);
//
// and the plane path at the bottom of this file was its only caller.  Its own comment
// said the hull decode gets the LITERAL transcription "because vertex2Shape applies it to
// every hull vertex and the result feeds GJK, where a last-bit difference can flip a
// support-vertex choice" -- which concedes the two forms differ in the last bit, and then
// hands the different one to the plane path anyway.
//
// The plane path's engine origin is softbodyPrimitives.cu:268,
// `pInPlaneSpace = planeTransform.transformInv(vB)`, and PxTransform::transformInv is
// `q.rotateInv(input - p)` (PxTransform.h) -- the literal PxQuat::rotateInv, the same
// function the hull decode gets.  The difference is observable twice over: pInPlaneSpace.x
// gates contact existence through `cDistance >= pInPlaneSpace.x`, and it is written out as
// the penetration the solver consumes.  Of 63 rotateInv copies in this corpus the SPIR-V
// says 62 share one arithmetic DAG; this was the one that did not.
//
// PxQuat::rotateInv (PxQuat.h:324), transcribed term for term.
float3 cgQuatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// vertex2Shape (cudaNpCommon.h:270).  PxMeshScale is a NON-UNIFORM scale expressed in a
// rotated frame: rotate the vertex into that frame, scale it componentwise, rotate back.
// It is not `v * scale` unless the scale rotation is identity.
float3 cgVertex2Shape(float3 v, float3 scale, float4 rot)
{
    float3 temp = cgQuatRotate(rot, v);
    temp.x *= scale.x;
    temp.y *= scale.y;
    temp.z *= scale.z;
    return cgQuatRotateInv(rot, temp);
}

// PxQuat::getBasisVector0() — the rotated x axis.
float3 quatBasisVector0(float4 q)
{
    const float x2 = q.x * 2.0f, w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2,
                  (q.z * w2) + q.y * x2,
                  (-q.y * w2) + q.z * x2);
}

// ===========================================================================
// P8.6 — the SPHERE and CAPSULE branch of sbPrimitiveCollision
// (softbodyPrimitives.cu:470-568), on top of the shared GJK/EPA driver.
//
// The rigid becomes a POINT CLOUD and the tetrahedron the second one, and
// tetPrimitivesCollide2 does the rest.  Sphere and capsule are the two types whose cloud
// needs no convex-hull decode at all -- one vertex for a sphere, two for a capsule's segment
// endpoints -- which is why they landed first.  BOX and CONVEXMESH followed and share ONE
// branch: to this code a box is just a cooked eight-vertex hull, so both walk the same blob
// (uint4 header, float4 extents, float4 vertices) and both run prepareVertices.  The two
// differ in exactly two scalars -- the inner-sphere scale factor and nothing else.
//
// THE RADIUS IS NOT IN THE CLOUD.  A sphere's cloud is its CENTRE and a capsule's is its
// segment; the radius is folded into the contact distance instead (`cDistance + scale.x` for
// a sphere, `+ scale.y` for a capsule) and then subtracted back out of the penetration as
// `penOffset = -radius`.  That is how upstream keeps GJK working on point sets rather than
// on inflated hulls, and getting it wrong gives contacts that are exactly one radius off.
// A HULL HAS NO SUCH RADIUS: `radius` stays 0 for box and convex mesh, the contact distance
// is passed through unmodified, and the penetration needs no correction.
//
// NOTE which scale component is which: PxMeshScale.scale.x is a sphere's radius and a
// capsule's HALF-HEIGHT, while scale.y is a capsule's radius -- and `radius`, the value used
// for penOffset, is scale.y for BOTH (softbodyPrimitives.cu:477-481).  For a sphere the
// engine stores the same radius in x and y, so the two agree; the asymmetry is upstream's
// and is transcribed rather than tidied.
//
// THE WHOLE WARP, AS THE CUDA.  Until the cooperative rework the query ran on lane 0 with
// the serial driver (pxgSlangTetCollide: one thread simulating the warp in 32-element
// per-lane arrays) while 31 lanes waited.  It now runs on all 32 lanes through
// pxgSlangGjkWarp -- the same GJK and EPA, lane L holding what CUDA lane L holds, the
// clouds in groupshared like the CUDA's TetCollideScratch, each shuffle an exchange
// through groupshared -- and returns the same bits (npTetCollide_warpDifferential pins
// the two drivers against each other; softBodySimPrimitiveContactGenWarp pins this
// kernel against the serial driver).  The CUDA's 12-lane cooperative vertex load and
// its shuffled inner sphere are not reproduced: every lane reads the four corners and
// computes the inner sphere itself, the serial arithmetic on the same inputs, so the
// value is uniform.  Lane 0 then writes the one contact the pair can produce -- which is
// what writeVolumeContactFullWarp does anyway: it lets lane 0 take the atomic and
// broadcasts the index, and only lanes 0..3 store.
//
// Group-uniform by construction: every value the exchanges branch on is read back at a
// fixed lane, so all 32 threads run the same barriers.  The caller's per-pair barrier
// separates this pair's cloud writes from the previous pair's reads.
void sbGjkPrimitive(uint lane, PxgShape rigidShape, uint rigidCacheRef, uint softbodyId,
                    uint tetrahedronIdx, float cDistance, float restDist,
                    uint2 rigidId, uint64_t tetRef, Ptr<PxgSoftBodyGpu> softbodies)
{
    // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
    const uint64_t xf = gCG.transformCache + uint64_t(rigidCacheRef) * 32UL;
    const float4 rigidQ = Ptr<float4>(xf)[0];
    const float3 rigidP = float3(Ptr<float>(xf + 16UL)[0],
                                 Ptr<float>(xf + 20UL)[0],
                                 Ptr<float>(xf + 24UL)[0]);

    const uint4 tetIdx = Ptr<uint4>(softbodies[softbodyId].mTetIndices)[tetrahedronIdx];
    Ptr<float4> verts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);
    const float3 b0 = verts[tetIdx.x].xyz;
    const float3 b1 = verts[tetIdx.y].xyz;
    const float3 b2 = verts[tetIdx.z].xyz;
    const float3 b3 = verts[tetIdx.w].xyz;

    // The tet is cloud B, one corner per lane 0..3.
    if (lane < 4u)
        gwCloudPut(1u, lane, (lane == 0u) ? b0 : (lane == 1u) ? b1 : (lane == 2u) ? b2 : b3);

    float3 tetCenter;
    const float innerSphere = computeInnerSphere(b0, b1, b2, b3, tetCenter);

    uint nbA;
    float contactDist;
    float inSphereRadius0 = 0.0f;
    // `radius` is scale.y for a sphere and a capsule and ZERO for a hull — the hull carries
    // its own thickness in its vertices, so nothing is folded into the contact distance and
    // nothing has to be taken back out (softbodyPrimitives.cu:475-481).
    float radius = 0.0f;
    if (rigidShape.type == GEOM_SPHERE)
    {
        nbA = 1u;
        if (lane == 0u) gwCloudPut(0u, 0u, rigidP);
        contactDist = cDistance + rigidShape.scaleX;
        radius = rigidShape.scaleY;
    }
    else if (rigidShape.type == GEOM_CAPSULE)
    {
        nbA = 2u;
        const float3 wHalfHeight = cgQuatRotate(rigidQ, float3(rigidShape.scaleX, 0.0f, 0.0f));
        if (lane == 0u) gwCloudPut(0u, 0u, rigidP + wHalfHeight);
        if (lane == 1u) gwCloudPut(0u, 1u, rigidP - wHalfHeight);
        contactDist = cDistance + rigidShape.scaleY;
        radius = rigidShape.scaleY;
    }
    else   // GEOM_BOX or GEOM_CONVEXMESH — both are the same convex-hull blob
    {
        // The blob starts with one float4 that the collision code skips
        // (`convexPtr = hullPtr + sizeof(float4)`, softbodyPrimitives.cu:471); from there
        // it is a uint4 header, a float4 of extents, then the hull vertices as float4.
        // hullOrMeshOffset carries `size_t hullOrMeshPtr` verbatim — under this bridge a
        // CUdeviceptr IS a VkDeviceAddress, so the two halves reassemble into a BDA.
        const uint64_t hullPtr = uint64_t(rigidShape.hullOrMeshOffset.x) |
                                 (uint64_t(rigidShape.hullOrMeshOffset.y) << 32);
        const uint64_t convexPtr = hullPtr + 16UL;
        const uint4 header = Ptr<uint4>(convexPtr)[0];
        const float4 extents = Ptr<float4>(convexPtr + 16UL)[0];

        // header.x = polyData0_NbEdgesNbHullVerticesNbPolygons; getNbVerts = u16High(u32Low).
        uint nbVertices0 = (((header.x & 0xffffu) >> 8) & 0xffu);
        // CONVEX_MAX_VERTICES_POLYGONS (convexNpCommon.h:52) is 64, which is exactly
        // GJK_MAX_CLOUD; upstream asserts the bound, we clamp it so a corrupt header cannot
        // walk off the end of the cloud.
        nbVertices0 = min(nbVertices0, GJK_MAX_CLOUD);

        // extents.w is the hull's stored inner-sphere radius in VERTEX space.  A box is the
        // one type whose PxMeshScale is the half-extent itself, so the radius has to shrink
        // with the tightest axis; a convex mesh's scale is already baked into the blob and
        // upstream leaves the factor at 1 (softbodyPrimitives.cu:513-519).
        float minimumScale = 1.0f;
        if (rigidShape.type == GEOM_BOX)
            minimumScale = min(rigidShape.scaleZ, min(rigidShape.scaleX, rigidShape.scaleY));
        inSphereRadius0 = extents.w * minimumScale;

        // NO radius inflation for a hull.
        contactDist = cDistance;

        // prepareVertices (cudaNpCommon.h:315): vertex space -> shape space -> world, one
        // vertex per lane as the CUDA's `for (i = threadIdx.x; i < nb; i += WARP_SIZE)`.
        const float3 scale = float3(rigidShape.scaleX, rigidShape.scaleY, rigidShape.scaleZ);
        const float4 scaleRot = float4(rigidShape.scaleRotX, rigidShape.scaleRotY,
                                       rigidShape.scaleRotZ, rigidShape.scaleRotW);
        Ptr<float4> hullVerts = Ptr<float4>(convexPtr + 32UL);
        for (uint i = lane; i < nbVertices0; i += WARP)
        {
            // The clamp bounds the address the masked-off SIMD lanes form (P9.22).
            const float3 vv = hullVerts[min(i, max(nbVertices0, 1u) - 1u)].xyz;
            gwCloudPut(0u, i, cgQuatRotate(rigidQ, cgVertex2Shape(vv, scale, scaleRot)) + rigidP);
        }
        nbA = nbVertices0;
    }

    GjkOut o;
    uint path;
    const uint result = gwTetPrimitivesCollide2(lane, nbA, 4u, rigidP - tetCenter, contactDist,
                                                inSphereRadius0, innerSphere, o, path);

    // Both OVERLAP and CLOSE mean "touching" to the caller (softbodyPrimitives.cu:561).
    if (!(result == GJK_OVERLAP || result == GJK_CLOSE))
        return;
    if (lane != 0u)
        return;

    // writeVolumeContactFullWarp (softbodyPrimitives.cu, 5.12): the stored point is the
    // witness on the TET, the normal is GJK/EPA's direction unflipped, and the penetration is
    // the witness separation projected on it, less the radius that was folded in above.
    const float3 wA = o.closestPointA.xyz;
    const float3 wB = o.closestPointB.xyz;
    const float3 n = o.direction.xyz;
    const float pen = dot(wB - wA, n) - radius;

    uint index;
    InterlockedAdd(Ptr<uint>(gCG.w_totalContactCount)[0], 1u, index);
    if (index >= gCG.w_maxNumContacts)
        return;

    // writer.writeSortKey(index, rigidRef), then the point, normal/pen, pair info and
    // PxU16 material -- writeVolumeContact's store set.
    cgWriter().writeVolumeContact(index,
                                  float4(wB.x, wB.y, wB.z, restDist),
                                  float4(n.x, n.y, n.z, pen),
                                  dfNodeU64(rigidId), rigidShape.materialIndex, tetRef);
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

    // One workgroup per CUDA warp (see the header): globalWarpIdx = threadIdx.y +
    // blockIdx.x * blockDim.y is the group index, the stride gridDim.x * blockDim.y the
    // group count.
    {
        const uint globalWarpIdx = gid.x;
        const uint stride = WorkgroupCount().x;

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
            const uint64_t tetRef = PxgEncodeDeformableTetrahedron(softbodyId, tetrahedronIdx);

            // 5.12: a filter pair for this rigid and tetrahedron suppresses contact
            // generation between them.  The array comes from the softbody; dfFilterArray
            // substitutes the descriptor for a null array, because llvmpipe SPECULATES the
            // first load of a zero-trip binary search (bisected in P9.38: a null filter
            // address faulted here).
            if (findRigidVolumeFilter(
                    softBodyRigidVolumeFilters(gCG.softbodies + uint64_t(softbodyId) * PXG_SOFTBODY_STRIDE),
                    dfNodeU64(rigidId), rigidCacheRef, softbodyId, tetrahedronIdx))
                continue;

            // *** Everything that is not a PLANE goes through GJK/EPA, exactly as in the
            // .cu: sphere, capsule, box and convex mesh all become point clouds. ***
            if (rigidShape.type == GEOM_SPHERE || rigidShape.type == GEOM_CAPSULE ||
                rigidShape.type == GEOM_BOX || rigidShape.type == GEOM_CONVEXMESH)
            {
                sbGjkPrimitive(lane, rigidShape, rigidCacheRef, softbodyId, tetrahedronIdx,
                               cDistance, restDist, rigidId, tetRef, softbodies);
                continue;
            }

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
                const float3 pInPlaneSpace = cgQuatRotateInv(planeQ, wp - planeP);   // P9.37
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
                cgWriter().writeVolumeContact(index,
                                              float4(worldPoint.x, worldPoint.y, worldPoint.z, restDist),
                                              float4(worldNormal.x, worldNormal.y, worldNormal.z, pen),
                                              dfNodeU64(rigidId), rigidShape.materialIndex, tetRef);
            }
        }
    }

    // atomicMax(stackSizeNeededOnDevice, numMidphasePairsFound * sizeof(uint4)) — how the
    // engine learns to grow the collision stack for the next frame.
    if (gid.x == 0u && lane == 0u)
        InterlockedMax(Ptr<uint>(gCG.stackSizeNeededOnDevice)[0], numFound * 16u);
}
