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
// softBodyMeshMidphaseBda — sb_meshMidphaseGeneratePairsLaunch
// (gpunarrowphase/src/CUDA/softbodySoftbodyMidPhase.cu:1268, core at :979, launch
// PxgNarrowphaseCore.cpp:5035, SB_MESH_MIDPHASE), on the engine BDA ABI.
//
// Step 1 of TWO for softbody-vs-TRIANGLE-MESH — the pair that lets a FEM volume rest on cooked
// static geometry instead of on a primitive.  One warp per surface tetrahedron of the softbody
// walks the TRIANGLE MESH's BV32 tree and pushes its four closest triangles onto the shared
// midphase stack; step 2 (sb_meshContactGenLaunch) runs GJK/EPA on each pair.  The contacts go
// into the RIGID contact stream through PxgRigidDbContactWriter (5.12), so everything downstream --
// fem_reorderRigidContactsLaunch, sb_other_contact_remap_to_simLaunch,
// sb_rigidContactPrepareLaunch, sb_solveRigidSoftCollisionLaunch -- is already mapped.
//
// THE TRAVERSER IS THE ONE softBodyClothMidphaseBda ALREADY CARRIES, and that is a measured
// claim rather than a convenience.  Both call the same
// `TriangleLeafBoundMinMaxTraverser<BackFaceCull>` (softbodySoftbodyMidPhase.cu:117); the cloth
// path instantiates it `<false>` and this one `<true>`.  The template argument switches ONLY
// `satIntersect`'s internal `plane.distance(tet.centroid) < 0 -> PX_MAX_F32` early-out
// (:79-87), and the traverser body already applies exactly that test itself at :148 before
// calling in, for BOTH instantiations.  So `<true>` and `<false>` compute the same number here
// and the traversal is transcribed once.  (The cloth port's header records the other side of
// this: carrying the internal cull as well made the outer gate provably redundant and a mutant
// deleting it survived.  One gate, where the .cu puts it.)
//
// WHAT IS ACTUALLY DIFFERENT FROM THE CLOTH MIDPHASE:
//
//   1. THE TREE IS A COOKED BLOB, NOT A DESCRIPTOR.  The cloth hands over
//      mTriMeshData/mPosition_InvMass/mTriangleVertexIndices as three fields.  Here everything
//      hangs off `trimeshShape.hullOrMeshPtr` and has to be walked (readTriangleMesh,
//      triangleMesh.cuh:120):
//
//          uint4 header = (nbVerts, nbTris, nbBv32TreeNodes, nbPackedNodes)   @0
//          BV32DataPacked[header.w]                                          @16
//          float4  verts[header.x]                                           @16 + 1168*header.w
//          uint4   indices[header.y]                                         ... + 16*header.x
//
//      NOTE header.z is nbBv32TreeNodes and header.w is nbPackedNodes -- the node ARRAY is
//      sized by .w, not .z.  They differ, and using .z walks the vertex pointer into the middle
//      of the tree.
//
//   2. THE QUERY TETRAHEDRON IS TRANSFORMED INTO THE MESH'S *VERTEX* SPACE, not merely into its
//      local frame.  Two steps, and both matter for a scaled mesh:
//          world -> mesh local:   trimeshToWorld.transformInv(v) = q.rotateInv(v - p)
//          local -> vertex space: shape2Vertex(v, scale, rot) = rot.rotateInv(rot.rotate(v) / scale)
//      The tree's bounds and vertices are stored in vertex space, so skipping the second step
//      silently works for scale (1,1,1) and breaks for everything else.
//
//   3. THE CONTACT DISTANCE IS DIVIDED BY THE SMALLEST |scale| COMPONENT, because it is a world
//      distance being compared against vertex-space separations.  Upstream:
//      `contactDist = worldContactDist / trimeshScale.scale.abs().minElement()`.  The DIVISION
//      is by the minimum, so the band is CONSERVATIVE (never smaller than the true world band).
//
//   4. no cloth descriptor in the blob -> 80 bytes rather than 88.
//
// *** THE GRID.  READ THIS BEFORE CHANGING numthreads. ***
// The engine launches grid (SB_SBMIDPHASE = 1024, numTests, 1) with block
// (WARP_SIZE, MIDPHASE_WARPS_PER_BLOCK = 2, 1) and grid-strides over the softbody's tetrahedra:
//
//     const PxU32 globalWarpIdx = threadIdx.y + blockIdx.x * blockDim.y;   // [0, 2048)
//     const PxU32 NbWarps       = blockDim.y * gridDim.x;                  // 2048
//
// The bridge forwards gridDimX/Y/Z but DROPS blockDim, so a [numthreads(32,1,1)] group is ONE
// warp, cmIdx = gid.y.  The block's two warps used to be walked serially in one group; the
// launch site now (PxgNarrowphaseCore.cpp, PXG_NO_CUDA_TOOLKIT) launches one group per CUDA
// warp, (SB_SBMIDPHASE * 2, numTests, 1), so globalWarpIdx = gid.x and the stride is the group
// count WorkgroupCount().x -- the CUDA's 2048, and at any other grid still a stride that
// visits every element exactly once.
//
// DEVIATIONS from CUDA (recipe policy, each preserving the emitted pair SET and the atomic
// total): no 32-lane intrinsics (__ballot_sync -> groupshared InterlockedOr, minIndex -> a
// groupshared float[32] plus a strict-`<` scan from lane 0, which picks the same lowest-lane
// minimum); the per-warp femMidphaseScratch is dropped and every lane reads the addresses
// directly; PxU8 hint arrays are read through the containing 32-bit word with `& ~uint64_t(3)`
// (writing `& ~3UL` MISCOMPILES -- Slang evaluates the complement in 32 bits and clears the top
// half of the device address).
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): isValidTet's signed volume and every plane
// product in makePlane / minProject3 are mul-feeding-add chains that would otherwise fuse to
// FMAs and break parity with the -ffp-contract=off CPU reference.
// ===========================================================================
module softBodyMeshMidphaseBda;

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu

// PxgNpKernelIndices.h:32 / PxgSoftBodyCoreKernelIndices.h:55.
static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
static const uint SB_SBMIDPHASE = 1024u;

static const uint WARP = 32u;
static const float PX_MAX_F32 = 3.4028235e38f;

// Gu::BV32DataPacked: 1168 B — mMin[32] @0, mMax[32] @512, mData[32] @1024, mNbNodes @1152.
import pxgSlangBv32;   // P8.25: the BV32 reader, shared

// femMidphaseScratch.sBv32Nodes[192] — a 6-deep BV32 tree's worst-case pending set.
static const uint STACK_SIZE = 192u;

// The traverser keeps at most four triangles per query tetrahedron
// (softbodySoftbodyMidPhase.cu:1096, `const PxU32 MaxContacts = 4`).
static const uint MAX_CONTACTS = 4u;

static const uint GEOM_TETRAHEDRONMESH = 7u;

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA pack (PxgNarrowphaseCore.cpp:5037):
// { PxReal toleranceLength, cmInputs*, transformCache*, bounds*, contactDistance*, gpuShapes*,
//   softbodies*, PxU32 stackSizeBytes, stackPtr*, midphasePairsNum* }.
// Natural alignment: 4 @0, six pointers @8..@48, PxU32 @56, then a pointer realigned to @64 and
// @72 -> 80-byte blob.  ABI row {4,8,8,8,8,8,8,4,8,8}.
// ---------------------------------------------------------------------------
struct SbMeshMidphaseParams
{
    float toleranceLength;          // @0   (ABI parity only — the core never reads it)
    uint _pad4;                     // @4
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16
    uint64_t bounds;                // @24  (ABI parity only)
    uint64_t contactDistance;       // @32
    uint64_t gpuShapes;             // @40
    uint64_t softbodies;            // @48
    uint stackSizeBytes;            // @56
    uint _pad60;                    // @60
    uint64_t stackPtr;              // @64
    uint64_t midphasePairsNum;      // @72 -> 80
}

[[vk::push_constant]] ConstantBuffer<SbMeshMidphaseParams> gMm;

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
// BV32DataPacked field readers (explicit byte offset through Ptr<>).
// ---------------------------------------------------------------------------

// A PxU8 array read through the containing 32-bit word.  See the header on `& ~uint64_t(3)`.
uint byteAt(uint64_t base, uint idx)
{
    const uint64_t addr = base + uint64_t(idx);
    const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
    const uint shift = uint(addr & 3UL) * 8u;
    return (word >> shift) & 0xffu;
}

// ---------------------------------------------------------------------------
// Geometry, transcribed from softBodyClothMidphaseBda (which took it from the .cu).
// ---------------------------------------------------------------------------
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

bool isValidTet(float3 v0, float3 v1, float3 v2, float3 v3)
{
    return dot(v1 - v0, cross(v2 - v0, v3 - v0)) > 1e-9f;
}

// PxPlane(p0, p1, p2): n = normalize(cross(p1-p0, p2-p0)), d = -p0.dot(n).
void makePlane(float3 p0, float3 p1, float3 p2, out float3 n, out float d)
{
    n = getNormalized(cross(p1 - p0, p2 - p0));
    d = -dot(p0, n);
}

// Gu::createPlane: flip so pd lies on the non-positive side.
void createPlane(float3 pa, float3 pb, float3 pc, float3 pd, out float3 n, out float d)
{
    makePlane(pa, pb, pc, n, d);
    if (dot(n, pd) + d > 0.0f) { n = -n; d = -d; }
}

float minProject3(float3 n, float d, float3 v0, float3 v1, float3 v2)
{
    return min(dot(n, v0) + d, min(dot(n, v1) + d, dot(n, v2) + d));
}

// satIntersect (softbodySoftbodyMidPhase.cu:79): the triangle plane term, then the four tet
// planes, with an early-out whenever sep exceeds the tolerance -- and the tolerance EVOLVES
// with the running best, so the traversal gets cheaper as it finds closer triangles.  The
// template's internal BackFaceCull is NOT here; the caller's gate already is it (see header).
float satIntersectTetTri(float3 pn[4], float pd[4],
                         float3 v0, float3 v1, float3 v2, float tolerance)
{
    float3 n; float d;
    makePlane(v0, v1, v2, n, d);
    float sep = minProject3(n, d, v0, v1, v2);
    if (sep > tolerance) return sep;
    sep = max(sep, minProject3(pn[0], pd[0], v0, v1, v2));
    if (sep > tolerance) return sep;
    sep = max(sep, minProject3(pn[1], pd[1], v0, v1, v2));
    if (sep > tolerance) return sep;
    sep = max(sep, minProject3(pn[2], pd[2], v0, v1, v2));
    if (sep > tolerance) return sep;
    sep = max(sep, minProject3(pn[3], pd[3], v0, v1, v2));
    return sep;
}

// PxQuat::rotate / rotateInv (PxQuat.h:310, :324), literal.
float3 mmQuatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}
float3 mmQuatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// shape2Vertex(v, scale, rot) = rot.rotateInv(rot.rotate(v) / scale)  (cudaNpCommon.h:285).
// DIVIDES by the scale -- it is the inverse of vertex2Shape, which multiplies.
float3 mmShape2Vertex(float3 v, float3 scale, float4 rot)
{
    float3 temp = mmQuatRotate(rot, v);
    temp.x /= scale.x;
    temp.y /= scale.y;
    temp.z /= scale.z;
    return mmQuatRotateInv(rot, temp);
}

// LoadShapePairWarp<eTETRAHEDRONMESH, eTRIANGLEMESH>: the softbody is whichever half of the
// pair IS the tetrahedron mesh; the triangle mesh is the other.
void loadSoftbodyTrimeshPair(uint64_t cmInputsAddr, uint64_t shapesAddr, uint cmIdx,
                             out PxgShape sbShape, out uint sbCacheRef,
                             out PxgShape triShape, out uint triCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(cmInputsAddr)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(shapesAddr);
    uint sRefSb = cin.shapeRef0, cRefSb = cin.transformCacheRef0;
    uint sRefTri = cin.shapeRef1, cRefTri = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != GEOM_TETRAHEDRONMESH)
    {
        sRefSb = cin.shapeRef1; cRefSb = cin.transformCacheRef1;
        sRefTri = cin.shapeRef0; cRefTri = cin.transformCacheRef0;
    }
    sbShape = shapes[sRefSb]; sbCacheRef = cRefSb;
    triShape = shapes[sRefTri]; triCacheRef = cRefTri;
}

// ---------------------------------------------------------------------------
// Warp emulation scratch: one group == one warp, so a single slot each.
// ---------------------------------------------------------------------------
groupshared int sMmStack[STACK_SIZE];   // bv32 to-be-expanded node stack
groupshared uint sMmBallot;             // __ballot_sync mask
groupshared float sMmBest[WARP];        // minIndex reduction over the lane bests
groupshared uint sMmIndex0;             // atomicAdd result broadcast

uint mmBallot(bool pred, uint lane)
{
    if (lane == 0u) sMmBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sMmBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sMmBallot;
}

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void sb_meshMidphaseGeneratePairsLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint cmIdx = gid.y;                      // blockIdx.y — one pair per y-block

    PxgShape sbShape, triShape;
    uint sbCacheRef, triCacheRef;
    loadSoftbodyTrimeshPair(gMm.cmInputs, gMm.gpuShapes, cmIdx,
                            sbShape, sbCacheRef, triShape, triCacheRef);

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gMm.softbodies);
    const uint softbodyId = sbShape.particleOrSoftbodyId;

    // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
    const uint64_t xf = gMm.transformCache + uint64_t(triCacheRef) * 32UL;
    const float4 triQ = Ptr<float4>(xf)[0];
    const float3 triP = float3(Ptr<float>(xf + 16UL)[0],
                               Ptr<float>(xf + 20UL)[0],
                               Ptr<float>(xf + 24UL)[0]);

    const float3 triScale = float3(triShape.scaleX, triShape.scaleY, triShape.scaleZ);
    const float4 triScaleRot = float4(triShape.scaleRotX, triShape.scaleRotY,
                                      triShape.scaleRotZ, triShape.scaleRotW);

    // ---- the cooked triangle-mesh blob (readTriangleMesh, triangleMesh.cuh:120) ----
    const uint64_t meshPtr = uint64_t(triShape.hullOrMeshOffset.x) |
                             (uint64_t(triShape.hullOrMeshOffset.y) << 32);
    const uint4 meshHeader = Ptr<uint4>(meshPtr)[0];   // nbVerts, nbTris, nbBv32Nodes, nbPacked
    const uint64_t treeAddr = meshPtr + 16UL;
    // .w is nbPackedNodes and sizes the node ARRAY.  .z is a different count — see the header.
    const uint64_t vertsAddr = treeAddr + uint64_t(meshHeader.w) * BV32_STRIDE;
    const uint64_t indicesAddr = vertsAddr + uint64_t(meshHeader.x) * 16UL;
    Ptr<float4> triVerts = Ptr<float4>(vertsAddr);
    Ptr<uint4> triIndices = Ptr<uint4>(indicesAddr);

    // A world contact distance compared against VERTEX-space separations: divide by the
    // smallest |scale| component so the band is never smaller than the true world one.
    const float worldContactDist = Ptr<float>(gMm.contactDistance)[sbCacheRef] +
                                   Ptr<float>(gMm.contactDistance)[triCacheRef];
    const float3 absScale = abs(triScale);
    const float minScale = min(absScale.x, min(absScale.y, absScale.z));
    const float contactDist = worldContactDist / minScale;

    const uint nbTets = softbodies[softbodyId].mNumTets;
    Ptr<uint4> tetIndices = Ptr<uint4>(softbodies[softbodyId].mTetIndices);
    Ptr<float4> tetVerts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);
    const uint64_t surfaceHintBase = softbodies[softbodyId].mTetMeshSurfaceHint;

    const uint stackCapacity = gMm.stackSizeBytes / 16u;

    // One group per CUDA warp (see the header on the grid): globalWarpIdx is the group
    // index and the stride the group count, i.e. blockDim.y * gridDim.x.
    {
    const uint globalWarpIdx = gid.x;

    for (uint tetrahedronIdx = globalWarpIdx; tetrahedronIdx < nbTets; tetrahedronIdx += WorkgroupCount().x)
    {
        GroupMemoryBarrierWithGroupSync();   // groupshared scratch is reused per query

        // P9.49: safe ADDRESS for the reads this platform issues even when the loop
        // condition admits no lane (see femClothPsMidphaseBda, where it faulted).
        // BOTH reads below need it -- the surface hint is read before the `continue`,
        // so it is the first one out of range, not the index fetch.
        const uint safeTet = min(tetrahedronIdx, max(nbTets, 1u) - 1u);

        // Interior tetrahedra never generate contacts.
        if (byteAt(surfaceHintBase, safeTet) == 0u)
            continue;

        const uint4 tetIdx = tetIndices[safeTet];

        // World -> mesh local -> mesh VERTEX space, all four corners.  Both steps, see header.
        const float3 t0 = mmShape2Vertex(mmQuatRotateInv(triQ, tetVerts[tetIdx.x].xyz - triP), triScale, triScaleRot);
        const float3 t1 = mmShape2Vertex(mmQuatRotateInv(triQ, tetVerts[tetIdx.y].xyz - triP), triScale, triScaleRot);
        const float3 t2 = mmShape2Vertex(mmQuatRotateInv(triQ, tetVerts[tetIdx.z].xyz - triP), triScale, triScaleRot);
        const float3 t3 = mmShape2Vertex(mmQuatRotateInv(triQ, tetVerts[tetIdx.w].xyz - triP), triScale, triScaleRot);

        // computeTetBoundWithTetrahedron: the isValidTet gate, the AABB, and the four outward
        // planes plus the centroid the SAT term needs.
        if (!isValidTet(t0, t1, t2, t3))
            continue;

        const float3 tMin = min(min(t0, t1), min(t2, t3));
        const float3 tMax = max(max(t0, t1), max(t2, t3));

        float3 pn[4]; float pd[4];
        createPlane(t0, t1, t2, t3, pn[0], pd[0]);
        createPlane(t0, t1, t3, t2, pn[1], pd[1]);
        createPlane(t0, t2, t3, t1, pn[2], pd[2]);
        createPlane(t1, t2, t3, t0, pn[3], pd[3]);
        // constructTetrahedron's centroid is (a + b + c) / 3 — the first THREE corners, not all
        // four.  Transcribed, not corrected.
        const float3 centroid = (t0 + t1 + t2) * (1.0f / 3.0f);

        // --- TriangleLeafBoundMinMaxTraverser per-lane state ---
        float bestDist = PX_MAX_F32;
        uint bestPrimIndex = 0xffffffffu;

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
                uint primitiveIndex = 0xffffffffu;
                if (lane < nbPrimitives)
                    primitiveIndex = lane + primStartIndex;
                // P9.28: the `!= 0xffffffff` guard below is PER-LANE, so this platform turns it
                // into a mask and issues the reads under it for the EXCLUDED lanes too -- at
                // primStartIndex + 0xffffffff, sixty-four gigabytes out for a uint4 array.
                // `safePrim` is the ADDRESS those reads use; the sentinel still decides the
                // VALUE, so nothing observable changes.  Traced from a reproducible sporadic
                // SIGSEGV, not from reading.
                const uint safePrim = min(primitiveIndex,
                                          primStartIndex + max(nbPrimitives, 1u) - 1u);

                if (primitiveIndex != 0xffffffffu)
                {
                    const uint4 triIdx = triIndices[safePrim];
                    const float3 v0 = triVerts[triIdx.x].xyz;
                    const float3 v1 = triVerts[triIdx.y].xyz;
                    const float3 v2 = triVerts[triIdx.z].xyz;

                    float3 tn; float td;
                    makePlane(v0, v1, v2, tn, td);

                    // The front-face gate (softbodySoftbodyMidPhase.cu:148).  Both engine
                    // instantiations sit behind it, which is what makes satIntersect's own
                    // BackFaceCull dead code for this call site.
                    if (dot(tn, centroid) + td >= 0.0f)
                    {
                        const float separation =
                            satIntersectTetTri(pn, pd, v0, v1, v2, min(bestDist, contactDist));
                        if (separation < bestDist)
                        {
                            bestDist = separation;
                            bestPrimIndex = primitiveIndex;
                        }
                    }
                }
            }
            else
            {
                // intersectBoxFullWarp: the per-axis AABB-vs-AABB displacement.
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
                {
                    float3 dsp;
                    dsp.x = (mn.x > tMax.x) ? (mn.x - tMax.x) : ((mx.x < tMin.x) ? (mx.x - tMin.x) : 0.0f);
                    dsp.y = (mn.y > tMax.y) ? (mn.y - tMax.y) : ((mx.y < tMin.y) ? (mx.y - tMin.y) : 0.0f);
                    dsp.z = (mn.z > tMax.z) ? (mn.z - tMax.z) : ((mx.z < tMin.z) ? (mx.z - tMin.z) : 0.0f);
                    intersect = dot(dsp, dsp) < contactDist * contactDist;
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

                const uint mask = mmBallot(isValidNode, lane);
                const uint offset = countbits(mask & ((1u << lane) - 1u));
                const uint validCount = countbits(mask);
                if (isValidNode)
                    sMmStack[nbStack + offset] = int(currentIndex);
                nbStack += validCount;
            }

            GroupMemoryBarrierWithGroupSync();
            if (nbStack > 0u)
            {
                nbStack -= 1u;
                currentNodeIndex = uint(sMmStack[nbStack]);
            }
            else
                hasNode = false;
        }

        // --- the top-4 extraction (softbodySoftbodyMidPhase.cu:1094) ---------------
        GroupMemoryBarrierWithGroupSync();
        sMmBest[lane] = bestDist;
        const uint mask = mmBallot(bestDist < contactDist, lane);
        uint count = countbits(mask);
        if (count > MAX_CONTACTS) count = MAX_CONTACTS;

        if (count == 0u)
            continue;

        // ONE atomicAdd of the whole count, broadcast to the warp.
        if (lane == 0u)
        {
            uint idx;
            InterlockedAdd(Ptr<uint>(gMm.midphasePairsNum)[0], count, idx);
            sMmIndex0 = idx;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint index = sMmIndex0;

        float bestSep = -1.0f;
        for (uint i = 0u; i < count && bestSep < contactDist; ++i)
        {
            // minIndex: the warp argmin, lowest lane winning ties.
            uint winnerLane = 0u;
            float best = PX_MAX_F32;
            for (uint j = 0u; j < WARP; ++j)
                if (sMmBest[j] < best) { best = sMmBest[j]; winnerLane = j; }
            bestSep = best;

            GroupMemoryBarrierWithGroupSync();
            if (winnerLane == lane && sMmBest[lane] < contactDist)
            {
                if ((index + i) < stackCapacity)
                    Ptr<uint4>(gMm.stackPtr)[index + i] =
                        uint4(cmIdx, tetrahedronIdx, bestPrimIndex, 0u);
                sMmBest[lane] = PX_MAX_F32;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    }
}
