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
//
// ===========================================================================
// femClothMeshContactGenBda — cloth_meshContactGenLaunch
// (gpunarrowphase/src/CUDA/femClothPrimitives.cu:3749, core
// staticTriangleVsClothCollision at :3605, launch PxgNarrowphaseCore.cpp:6669 inside
// testSDKFemClothTrimesh, CLOTH_MESH_CG), on the engine BDA ABI.
//
// P8.17 step 2 of TWO — the consumer step 1's midphase had none.  One THREAD per midphase pair
// clips the cloth triangle against the mesh triangle, takes the deepest surviving point, and
// writes a rigid-vs-deformable contact.  Everything downstream is already mapped: the contacts
// go into the RIGID contact stream through PxgRigidDbContactWriter (5.12), so
// cloth_rigidContactPrepare -> query -> cloth_solveRigidClothCollision carry them.
//
// THE CORE IS A LIFT, NOT A NEW TRANSCRIPTION.  `fcStaticTriCloth.slang` already ports exactly
// this routine for the descriptor-set engine (its MODE 0), with its own fixture
// (testFcStaticTriCloth.cpp) covering the trim pipeline's sign-change bookkeeping, the
// deepest-point averaging band and the filter skip against a CPU transcription.  The clipping code below is that port, verbatim; what is NEW here is the BDA
// plumbing around it — the 176-byte push blob, the cooked-blob walk, the PxgFEMCloth descriptor
// reads and the pair-stack iteration, all of which `fcStaticTriCloth`'s header explicitly lists
// as "engine glue not ported".  Two copies of the clipping pipeline is a cost this tranche pays
// knowingly: the descriptor-set shader is a unit-test harness with a case-array interface and
// cannot be imported by a push-constant kernel without dragging its bindings along.
//
// THE ARG PACK IS **NOT** sb_meshContactGenLaunch's, even though both end in an 80-byte
// writer.  The middle differs (5.12; both lost the filterPairs / nbFilterPairs pair):
//
//   softbody: ..., shapeToRigidRemapTable*, stackPtr*, midphasePairsNum*, PxU32 stackSizeBytes,
//             stackSizeNeeded*, writer
//   cloth:    ..., shapeToRigidRemapTable*, PxU32 stackSizeBytes, stackPtr*, midphasePairsNum*,
//             stackSizeNeeded*, writer
//
// `stackSizeBytes` moves from @80 to @64 and the two stack pointers shift with it.  Reading one
// pack with the other's layout hands the kernel a byte count where it expects a device pointer.
//
// WHAT DIFFERS FROM THE SOFTBODY SIBLING'S CONTACT GEN, beyond the pack:
//   1. no GJK/EPA.  The cloth path is a POLYGON CLIP: the cloth triangle is trimmed by the mesh
//      triangle's three edge planes and its two +-cDistance slabs, and the contact is the deepest
//      surviving corner (or the average of the corners within 0.01*cDistance of it);
//   2. the transform runs the OTHER WAY.  Step 1's midphase brings the cloth into the mesh's
//      vertex space; this kernel brings the MESH TRIANGLE into the CLOTH's local frame --
//      `vertex2Shape` (which MULTIPLIES by the scale) then
//      `clothTransform.transformInv(trimeshToWorld)`.  The cloth positions are used untransformed;
//   3. the filter (5.12) is one (rigid, TRIANGLE) search of the cloth's
//      mRigidSurfaceFilterPairs -- 5.6.1 needed all three cloth vertices filtered;
//   4. the write is PxgRigidDbContactWriter::writeSurfaceContact (5.12) — the barycentric,
//      the pair info (rigid node, PxgEncodeDeformableTriangle), the PxU16 material and the
//      full rigid sort key.  (5.6.1's writeRigidVsDeformableContact32 wrote only the 32-bit
//      contactSortedByX lane and left the u64 keys alone.)
//
// *** THE GRID.  READ THIS BEFORE CHANGING numthreads. ***
// The engine launches grid (2048, 1, 1) with block (128, 1, 1) and grid-strides:
//
//     globalThreadIdx = threadIdx.x + blockIdx.x * blockDim.x;   // [0, 262144)
//     for (i = globalThreadIdx; i < l; i += gridDim.x * blockDim.x)   // stride 262144
//
// with `l = ceil(numPairs / 32) * 32` — rounded UP to a warp multiple so every lane of every
// partially-filled warp still reaches the ballot.  The bridge drops blockDim, so a
// [numthreads(32,1,1)] group is ONE warp; the block's FOUR warps used to be walked serially in
// a group-uniform `sub` loop.  The launch site now (PxgNarrowphaseCore.cpp,
// PXG_NO_CUDA_TOOLKIT) launches one group per CUDA warp, 2048 * 4 groups, so group g is CUDA
// warp g -- threads g * 32 + lane, the CUDA's `threadIdx.x + blockIdx.x * blockDim.x` with
// warp g = blockIdx.x * 4 + threadIdx.x / 32 -- and the stride is the thread count actually
// launched, WorkgroupCount().x * 32 (the CUDA 262144 at that grid; at any other grid still
// every pair exactly once, as the start is below the stride).  The loop is driven on the warp
// BASE rather than on the per-lane index: `base` and the stride are both multiples of 32 and
// `l` is too, so `base + lane < l` is the same for every lane and the ballot's barriers stay
// uniform.  A block-scaled start with a group-count stride would double-visit pairs.
//
// DEVIATIONS from CUDA (recipe policy, each preserving the emitted contact SET and the atomic
// total): globalScanExclusiveSingleWarp -> a groupshared InterlockedOr ballot plus
// countbits(mask & lanemask_lt) and a single lane-0 reservation; every descriptor read goes
// through Ptr<> instead of a cached __shared__ block.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the plane distances, the trim pipeline's
// lerps and the barycentric solve are mul-feeding-add chains compared bit-exactly against the
// -ffp-contract=off CPU reference.
// ===========================================================================
module femClothMeshContactGenBda;

import pxgSlangCommon;   // quatRotate / quatRotateInv / PxgTransform28
import pxgSlangDefFilter;   // 5.12: findRigidSurfaceTriangleFilter
import pxgSlangDeformableRef;       // PxgEncodeDeformableTriangle
import pxgSlangDeformableWriters;   // PxgRigidDbContactWriterGpu
import pxgSlangFEMClothDesc;   // PxgFEMClothGpu
import pxgSlangClothTriContact;  // P9.27b: staticTriangleVsClothCollision + its clipping

static const float PX_NORMALIZATION_EPSILON = 1e-20f;

// Gu::BV32DataPacked stride, for the cooked-blob walk.
static const uint64_t BV32_STRIDE = 1168UL;

// The engine's launch shape (PxgNarrowphaseCore.cpp:6694).
static const uint CLOTH_CG_WARPS_PER_BLOCK = 4u;    // blockDim.x 128 / WARP_SIZE 32

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the 5.12 CUDA pack:
// { PxReal toleranceLength, cmInputs*, transformCache*, contactDistance*, restDistance*,
//   gpuShapes*, femClothes*, shapeToRigidRemapTable*, PxU32 stackSizeBytes, stackPtr*,
//   midphasePairsNum*, stackSizeNeededOnDevice*, PxgRigidDbContactWriter writer }.
// 4 @0, seven pointers @8..@56, PxU32 @64, a pointer realigned to @72, @80, @88, then the
// 80-byte writer at @96 (paramAlign(80) = 16; 96 is already 16-aligned) -> 176-byte blob.
// ABI row {4,8,8,8,8,8,8,8,4,8,8,8,80}.
// ---------------------------------------------------------------------------
struct ClothMeshCgParams
{
    float toleranceLength;          // @0   (ABI parity only — never read)
    uint _pad4;                     // @4
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16
    uint64_t contactDistance;       // @24
    uint64_t restDistance;          // @32
    uint64_t gpuShapes;             // @40
    uint64_t femClothes;            // @48
    uint64_t shapeToRigidRemapTable;// @56  PxNodeIndex* (u64 per transform-cache slot)
    uint stackSizeBytes;            // @64
    uint _pad68;                    // @68
    uint64_t stackPtr;              // @72  uint4 pairs from step 1
    uint64_t midphasePairsNum;      // @80
    uint64_t stackSizeNeededOnDevice; // @88
    // --- PxgRigidDbContactWriter, by value (PxgDeformableContactWriters.h, 5.12) ---
    uint64_t w_outPoint;                  // @96  float4*
    uint64_t w_outNormalPen;              // @104 float4*
    uint64_t w_outBarycentric;            // @112 float4*
    uint64_t w_outContactInfo;            // @120 PxgDeformablePairInfo*
    uint64_t w_outRigidMaterialIndex;     // @128 PxU16*
    uint64_t w_totalContactCount;         // @136 PxU32*
    uint64_t w_contactByRigid;            // @144 PxU64*
    uint64_t w_tempContactByRigid;        // @152 PxU32*
    uint64_t w_contactIndexSortedByRigid; // @160 PxU32*
    uint w_maxNumContacts;                // @168 PxU32
    uint _pad172;                         // @172 -> 176
}

[[vk::push_constant]] ConstantBuffer<ClothMeshCgParams> gCg;

// PxgContactManagerInput (PxgContactManager.h): four PxU32.
struct PxgContactManagerInput { uint shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; }

// PxgShape (PxgConvexConvexShape.h, 48 B) comes from pxgSlangCommon rather than a local copy —
// this shader already imports that module for quatRotate / PxgTransform28, and a second
// declaration is an ambiguous reference rather than a shadow.


// ---------------------------------------------------------------------------
// The plane/barycentric helpers, the clipping pipeline and
// staticTriangleVsClothCollision now live in pxgSlangClothTriContact (P9.27b),
// because the HEIGHTFIELD contact gen calls the same upstream __device__ function
// and a second copy is how two ports start disagreeing -- see P9.27a for the one
// that did.  What stays here is this kernel's own glue: the pair-stack walk, the
// shape routing, the filter and the writer.
// ---------------------------------------------------------------------------
PxgRigidDbContactWriterGpu cgWriter()
{
    PxgRigidDbContactWriterGpu w;
    w.outPoint = gCg.w_outPoint;                 w.outNormalPen = gCg.w_outNormalPen;
    w.outBarycentric = gCg.w_outBarycentric;     w.outContactInfo = gCg.w_outContactInfo;
    w.outRigidMaterialIndex = gCg.w_outRigidMaterialIndex;
    w.totalContactCount = gCg.w_totalContactCount;
    w.contactByRigid = gCg.w_contactByRigid;     w.tempContactByRigid = gCg.w_tempContactByRigid;
    w.contactIndexSortedByRigid = gCg.w_contactIndexSortedByRigid;
    w.maxNumContacts = gCg.w_maxNumContacts;
    return w;
}

// ---------------------------------------------------------------------------
// Warp emulation scratch: one group == one warp.
// ---------------------------------------------------------------------------
groupshared uint sCgBallot;      // __ballot_sync mask
groupshared uint sCgScanStart;   // the single lane-0 reservation, broadcast

uint cgBallot(bool pred, uint lane)
{
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) sCgBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sCgBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sCgBallot;
}

// ===========================================================================
// 5.12: findRigidSurfaceTriangleFilter(cloth->mRigidSurfaceFilterPairs,
// cloth->mNumRigidSurfaceFilterPairs, rigidId, trimeshCacheRef, clothId, triangle).
// dfFilterArray substitutes the cloth descriptor for a null array (count 0): llvmpipe
// speculates the first load of a zero-trip `while (l < r)`, see pxgSlangDefFilter.
bool clothTriangleFiltered(uint clothId, uint2 rigidId, uint rigidCacheRef, uint triangleId)
{
    return findRigidSurfaceTriangleFilter(clothRigidSurfaceFilters(gCg.femClothes + uint64_t(clothId) * PXG_FEMCLOTH_STRIDE),
                                          dfNodeU64(rigidId), rigidCacheRef, clothId, triangleId);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_meshContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    const uint stackSize = gCg.stackSizeBytes / 16u;
    const uint numMidphasePairsFound = Ptr<uint>(gCg.midphasePairsNum)[0];
    const uint numPairs = min(numMidphasePairsFound, stackSize);
    // l = ceil(numPairs / WARP_SIZE) * WARP_SIZE — every lane of a partial warp still ballots.
    const uint l = ((numPairs + 31u) / 32u) * 32u;

    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gCg.femClothes);
    Ptr<float> contactDistance = Ptr<float>(gCg.contactDistance);
    Ptr<float> restDistance = Ptr<float>(gCg.restDistance);
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gCg.gpuShapes);

    // One group per CUDA warp (see the header on the grid): the warp's first thread is
    // gid.x * 32 and the stride the launched thread count, gridDim.x * blockDim.x.
    {
    const uint warpBase = gid.x * 32u;

    // `base`, the stride and `l` are all multiples of 32, so this condition is the same for
    // every lane and the ballot's barriers below stay group-uniform.
    for (uint base = warpBase; base < l; base += WorkgroupCount().x * 32u)
    {
        const uint i = base + lane;

        bool hasContact = false;
        uint cmIdx = 0u;
        uint clothId = 0u;
        uint clothTriangleIdx = 0u;
        uint2 rigidId = uint2(0u, 0u);
        uint rigidMaterialIndex = 0xFFFFFFFFu;
        FemContact c;
        c.position = float3(0, 0, 0);
        c.normal = float3(0, 0, 0);
        c.separation = 0.0f;
        c.barycentric = float4(0, 0, 0, 0);

        if (i < numPairs)
        {
            const uint4 curPair = Ptr<uint4>(gCg.stackPtr)[i];
            cmIdx = curPair.x;
            clothTriangleIdx = curPair.y;              // the CLOTH triangle
            const uint meshTriangleIdx = curPair.z;    // the MESH triangle

            const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gCg.cmInputs)[cmIdx];
            const PxgShape shape0 = shapes[cin.shapeRef0];
            const PxgShape shape1 = shapes[cin.shapeRef1];

            const float cDistance = contactDistance[cin.transformCacheRef0] +
                                    contactDistance[cin.transformCacheRef1];

            // Both halves are eTRIANGLEMESH; the CLOTH is the one carrying a body id.
            PxgShape clothShape = shape0, trimeshShape = shape1;
            uint clothCacheRef = cin.transformCacheRef0, trimeshCacheRef = cin.transformCacheRef1;
            if (shape0.particleOrSoftbodyId == 0xffffffffu)
            {
                clothShape = shape1;   trimeshShape = shape0;
                clothCacheRef = cin.transformCacheRef1; trimeshCacheRef = cin.transformCacheRef0;
            }
            rigidMaterialIndex = trimeshShape.materialIndex;
            rigidId = Ptr<uint2>(gCg.shapeToRigidRemapTable)[trimeshCacheRef];   // PxNodeIndex
            clothId = clothShape.particleOrSoftbodyId;

            const uint4 clothVertInd = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices_ownedEdges)[clothTriangleIdx];

            // A filter pair for this rigid and triangle suppresses contact generation between them.
            const bool skipTriangle = clothTriangleFiltered(clothId, rigidId, trimeshCacheRef, clothTriangleIdx);

            if (!skipTriangle)
            {
                // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
                const uint64_t xfTri = gCg.transformCache + uint64_t(trimeshCacheRef) * 32UL;
                const float4 triQ = Ptr<float4>(xfTri)[0];
                const float3 triP = float3(Ptr<float>(xfTri + 16UL)[0],
                                           Ptr<float>(xfTri + 20UL)[0],
                                           Ptr<float>(xfTri + 24UL)[0]);
                const uint64_t xfCloth = gCg.transformCache + uint64_t(clothCacheRef) * 32UL;
                const float4 clothQ = Ptr<float4>(xfCloth)[0];
                const float3 clothP = float3(Ptr<float>(xfCloth + 16UL)[0],
                                             Ptr<float>(xfCloth + 20UL)[0],
                                             Ptr<float>(xfCloth + 24UL)[0]);

                // The cooked triangle-mesh blob (readTriangleMesh, triangleMesh.cuh:120).
                // `.w` is nbPackedNodes and sizes the node ARRAY — `.z` is a different count.
                const uint64_t meshPtr = uint64_t(trimeshShape.hullOrMeshOffset.x) |
                                         (uint64_t(trimeshShape.hullOrMeshOffset.y) << 32);
                const uint4 meshHeader = Ptr<uint4>(meshPtr)[0];
                const uint64_t vertsAddr = meshPtr + 16UL + uint64_t(meshHeader.w) * BV32_STRIDE;
                const uint64_t indicesAddr = vertsAddr + uint64_t(meshHeader.x) * 16UL;

                const uint4 triVertInd = Ptr<uint4>(indicesAddr)[meshTriangleIdx];
                const float3 triV0 = Ptr<float4>(vertsAddr)[triVertInd.x].xyz;
                const float3 triV1 = Ptr<float4>(vertsAddr)[triVertInd.y].xyz;
                const float3 triV2 = Ptr<float4>(vertsAddr)[triVertInd.z].xyz;

                const float3 triScale = trimeshShape.scale();
                const float4 triScaleRot = trimeshShape.scaleRotation();

                // trimeshToClothTransform = clothTransform.transformInv(trimeshToWorld)
                const PxgTransform28 clothT = makeTransform28(clothQ, clothP);
                const PxgTransform28 meshT = makeTransform28(triQ, triP);
                const PxgTransform28 meshToCloth = transformInvTransform(clothT, meshT);

                const float3 localA0 = transformPt(meshToCloth.q(), meshToCloth.p(),
                                                   vertex2Shape(triV0, triScale, triScaleRot));
                const float3 localB0 = transformPt(meshToCloth.q(), meshToCloth.p(),
                                                   vertex2Shape(triV1, triScale, triScaleRot));
                const float3 localC0 = transformPt(meshToCloth.q(), meshToCloth.p(),
                                                   vertex2Shape(triV2, triScale, triScaleRot));

                Ptr<float4> clothPos = Ptr<float4>(clothes[clothId].mPosition_InvMass);
                const float3 a1 = clothPos[clothVertInd.x].xyz;
                const float3 b1 = clothPos[clothVertInd.y].xyz;
                const float3 c1 = clothPos[clothVertInd.z].xyz;

                hasContact = staticTriangleVsClothCollision(localA0, localB0, localC0, a1, b1, c1,
                                                            clothQ, clothP, cDistance, c);
            }
        }

        // globalScanExclusiveSingleWarp(hasContact, writer.totalContactCount).
        const uint mask = cgBallot(hasContact, lane);
        const uint offset = countbits(mask & ((1u << lane) - 1u));
        const uint validCount = countbits(mask);
        if (lane == 0u)
        {
            uint start = 0u;
            if (validCount > 0u)
                InterlockedAdd(Ptr<uint>(gCg.w_totalContactCount)[0], validCount, start);
            sCgScanStart = start;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint index = sCgScanStart + offset;

        if (hasContact)
        {
            const uint64_t rigidRef = dfNodeU64(rigidId);   // rigidId.getInd()
            const uint64_t triRef = PxgEncodeDeformableTriangle(clothId, clothTriangleIdx);
            cgWriter().writeSurfaceContact(index,
                                           float4(c.position, restDistance[cmIdx]),
                                           float4(c.normal, c.separation),
                                           rigidRef, rigidMaterialIndex, triRef, c.barycentric);
        }
    }
    }

    // The stack high-water mark, so the engine can grow the shared midphase stack.
    if (gid.x == 0u && lane == 0u)
        InterlockedMax(Ptr<uint>(gCg.stackSizeNeededOnDevice)[0], numMidphasePairsFound * 16u);
}
