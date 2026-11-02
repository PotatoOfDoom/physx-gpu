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
// softBodyMeshContactGenBda — sb_meshContactGenLaunch (gpunarrowphase/src/CUDA/
// softbodyPrimitives.cu:893, core `sbMeshCollision` at :669, launch
// PxgNarrowphaseCore.cpp:5128, SB_MESH_CG), on the engine BDA ABI.
//
// Step 2 of TWO for softbody-vs-TRIANGLE-MESH, and the last one: with this the pair runs end
// to end.  Step 1 pushed (cmIdx, tetIdx, triIdx) triples onto the midphase stack; this kernel
// runs GJK/EPA on each and appends the surviving contacts to the RIGID contact stream through
// PxgRigidDbContactWriter (5.12), so everything downstream -- fem_reorderRigidContactsLaunch,
// sb_other_contact_remap_to_simLaunch, sb_rigidContactPrepareLaunch,
// sb_solveRigidSoftCollisionLaunch -- was already mapped by the softbody-vs-rigid tranches.
//
// *** THE PUSH BLOB IS BYTE-IDENTICAL TO sb_primitiveContactGenLaunch'S. ***  Same thirteen
// arguments (5.12) in the same order and the same sizes, so the ABI row is the same 13 entries
// { 4,8,8,8,8,8,8,8,8,8,4,8,80 } and the same 176-byte blob.  That is upstream's doing,
// not a coincidence worth relying on blindly -- but it does mean the blob half of this port is
// already covered by the primitive path's fixture, and the new ground is the geometry.
//
// WHAT IS SHARED WITH THE PRIMITIVE (RIGID) CONTACT GEN, verbatim:
//   - the 5.12 rigid-volume filter, findRigidVolumeFilter over the softbody's
//     mRigidVolumeFilterPairs with the (rigid, tetrahedron) query;
//   - writeVolumeContactFullWarp's store set (PxgRigidDbContactWriter, 5.12): sort key, point,
//     normal/pen, PxgDeformablePairInfo and the PxU16 rigid material;
//   - the stored contact is the witness on the SECOND cloud, the normal is GJK/EPA's direction,
//     and the fourth component of the point is restDistance[cmIdx];
//   - the tet is cloud A in WORLD space with its own inner-sphere radius.
//
// WHAT IS NEW HERE:
//
//   1. THE TRIANGLE COMES OUT OF THE COOKED BLOB AND GOES TO WORLD.  Step 1 walked the same
//      blob; this kernel needs the vertex and index pools again
//      (header uint4 | BV32DataPacked[header.w] | float4 verts[header.x] | uint4 indices).
//      Each vertex then takes the FORWARD transform, the mirror image of step 1's:
//          vertex2Shape(v, scale, rot) = rot.rotateInv(rot.rotate(v) * scale)   <- MULTIPLIES
//          world = trimeshToWorld.rotate(that) + p
//      Step 1 divides by the scale, this one multiplies.  Getting them the same way round
//      leaves the midphase and the contact gen disagreeing about where the triangle is, which
//      shows up as contacts that the midphase found and GJK then rejects.
//
//   2. THE FRONT-FACE GATE IS ON GJK'S DIRECTION, NOT ON THE TET CENTROID.  After the collide:
//          intersect = (OVERLAP || CLOSE) && dot(gjkOutput.direction, n) >= 0
//      *** AND `n` IS BUILT WITH THE CROSS PRODUCT THE OTHER WAY ROUND. ***  Upstream
//      (softbodyPrimitives.cu:873) computes
//          v20 = vB[2] - vB[0];  v21 = vB[1] - vB[0];  n = normalize(cross(v20, v21))
//      i.e. cross(v2 - v0, v1 - v0) -- the NEGATIVE of the usual winding normal.  Writing the
//      familiar cross(v1 - v0, v2 - v0) flips the gate and keeps exactly the contacts it should
//      drop.  Transcribed as upstream has it.
//
//   3. inSphereRadius1 = 1e-6f for the triangle, upstream's literal, not 0.
//
//   4. THE PAIR IDS ARE RIGID-SHAPED: ref0 is the PxNodeIndex from
//      shapeToRigidRemapTable[trimeshCacheRef] (a PxU64), ref1 the softbody tetrahedron.
//      The triangle mesh is a RIGID actor to the solver, so the contact is a rigid-vs-deformable
//      one and the material index comes from the TRIMESH shape.
//
//   5. THE FILTER IS SINGLE-LEVEL: one findRigidVolumeFilter(rigid, trimeshCacheRef, tet) and
//      out.  No wildcard passes -- those belong to the deformable-vs-deformable paths.
//
// *** ONE THREAD PER PAIR, NOT ONE WARP. ***  The CUDA spreads a single tet-vs-triangle query
// across 32 lanes (cooperative vertex loads, warp-wide GJK).  The Slang GJK
// (pxgSlangGjk / pxgSlangTetCollide) is scalar and already validated that way by every softbody
// contact-gen port before this one, so lane 0 runs the whole query and lanes 1..31 return.
// The work PARTITION is unchanged -- the same (block, warp) pair owns the same stack entries --
// so the emitted contact set and the atomic total are identical.
//
// THE GRID: SB_MESHCG = 1024 blocks of (WARP_SIZE, MIDPHASE_WARPS_PER_BLOCK = 2), gridDim.y = 1.
// The bridge drops blockDim, so the block's two warps used to be walked serially in one group.
// The launch site now (PxgNarrowphaseCore.cpp, PXG_NO_CUDA_TOOLKIT) launches one group per
// CUDA warp, 2048 groups, so the warp index is gid.x and the stride the group count
// WorkgroupCount().x -- the CUDA's `gridDim.x * blockDim.y` = 2048 at that grid, and at any
// other grid still every stack entry exactly once.
//
// THE TAIL: `atomicMax(stackSizeNeededOnDevice, numMidphasePairsFound * sizeof(uint4))` uses the
// UNCLAMPED count, so the host learns how much stack the frame actually wanted even when the
// loop was clamped to what it had.  Dropping the tail silently pins the stack at its current
// size and the next frame truncates again.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the GJK/EPA support and witness arithmetic
// and computeInnerSphere's plane distances are mul-feeding-add chains that would otherwise fuse
// to FMAs and break parity with the -ffp-contract=off CPU reference.
// ===========================================================================
module softBodyMeshContactGenBda;

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu
import pxgSlangDefFilter;   // 5.12: findRigidVolumeFilter
import pxgSlangDeformableRef;     // PxgEncodeDeformableTetrahedron
import pxgSlangDeformableWriters; // PxgRigidDbContactWriterGpu
import pxgSlangGjk;            // GjkCloud / GjkOut / GJK_OVERLAP / GJK_CLOSE / GJK_MAX_CLOUD
import pxgSlangTetCollide;     // tetPrimitivesCollide2 / computeInnerSphere

// PxgNpKernelIndices.h:32 / PxgSoftBodyCoreKernelIndices.h — SB_MESHCG.
static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
static const uint SB_MESHCG = 1024u;

static const uint GEOM_TETRAHEDRONMESH = 7u;


// Gu::BV32DataPacked, 1168 B — only the stride is needed here, to skip the node array.
static const uint64_t BV32_STRIDE = 1168UL;

// The triangle's inner-sphere radius, upstream's literal (softbodyPrimitives.cu:857).
static const float TRI_INNER_SPHERE = 0.000001f;

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the 5.12 CUDA pack, and, as the header notes, to
// sb_primitiveContactGenLaunch's.
// { PxReal tolerenceLength, cmInputs*, transformCache*, contactDistance*, restDistance*,
//   gpuShapes*, softbodies*, shapeToRigidRemapTable*, stackPtr*, midphasePairsNum*,
//   PxU32 stackSizeBytes, stackSizeNeededOnDevice*, PxgRigidDbContactWriter writer }
// -> 4 @0, nine pointers @8..@72, PxU32 @80, a pointer @88, and the 80-byte writer at @96
// -> 176-byte blob.  ABI row { 4,8,8,8,8,8,8,8,8,8,4,8,80 }.
// ---------------------------------------------------------------------------
struct SbMeshCgParams
{
    float tolerenceLength;          // @0    (ABI parity only — never read)
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

[[vk::push_constant]] ConstantBuffer<SbMeshCgParams> gMc;

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

PxgRigidDbContactWriterGpu mcWriter()
{
    PxgRigidDbContactWriterGpu w;
    w.outPoint = gMc.w_outPoint;                 w.outNormalPen = gMc.w_outNormalPen;
    w.outBarycentric = gMc.w_outBarycentric;     w.outContactInfo = gMc.w_outContactInfo;
    w.outRigidMaterialIndex = gMc.w_outRigidMaterialIndex;
    w.totalContactCount = gMc.w_totalContactCount;
    w.contactByRigid = gMc.w_contactByRigid;     w.tempContactByRigid = gMc.w_tempContactByRigid;
    w.contactIndexSortedByRigid = gMc.w_contactIndexSortedByRigid;
    w.maxNumContacts = gMc.w_maxNumContacts;
    return w;
}

// PxQuat::rotate / rotateInv (PxQuat.h:286, :324), literal.
float3 mcQuatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}
float3 mcQuatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// vertex2Shape(v, scale, rot) = rot.rotateInv(rot.rotate(v) * scale)  (cudaNpCommon.h:270).
// MULTIPLIES by the scale — the inverse of step 1's shape2Vertex, which divides.
float3 mcVertex2Shape(float3 v, float3 scale, float4 rot)
{
    float3 temp = mcQuatRotate(rot, v);
    temp.x *= scale.x;
    temp.y *= scale.y;
    temp.z *= scale.z;
    return mcQuatRotateInv(rot, temp);
}

float3 mcGetNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

// LoadShapePairWarp<eTETRAHEDRONMESH, eTRIANGLEMESH>: the softbody is whichever half of the
// pair IS the tetrahedron mesh; the triangle mesh is the other.
void loadSoftbodyTrimeshPair(uint cmIdx, out PxgShape sbShape, out uint sbCacheRef,
                             out PxgShape triShape, out uint triCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gMc.cmInputs)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gMc.gpuShapes);
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

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void sb_meshContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    const uint stackSize = gMc.stackSizeBytes / 16u;
    const uint numFound = Ptr<uint>(gMc.midphasePairsNum)[0];
    const uint numPairs = min(numFound, stackSize);

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gMc.softbodies);
    Ptr<float> contactDistance = Ptr<float>(gMc.contactDistance);
    Ptr<float> restDistance = Ptr<float>(gMc.restDistance);

    // Lane 0 runs the whole query (see the header); the other 31 have nothing to do.
    if (lane == 0u)
    {
    // One group per CUDA warp (see the header on the grid): globalWarpIdx is the group
    // index and the stride the group count, i.e. gridDim.x * blockDim.y.
    {
    const uint globalWarpIdx = gid.x;

    for (uint i = globalWarpIdx; i < numPairs; i += WorkgroupCount().x)
    {
        const uint4 curPair = Ptr<uint4>(gMc.stackPtr)[i];
        const uint cmIdx = curPair.x;
        const uint tetrahedronInd = curPair.y;
        const uint triangleInd = curPair.z;

        PxgShape sbShape, triShape;
        uint sbCacheRef, triCacheRef;
        loadSoftbodyTrimeshPair(cmIdx, sbShape, sbCacheRef, triShape, triCacheRef);

        const float cDistance = contactDistance[sbCacheRef] + contactDistance[triCacheRef];
        const uint softbodyId = sbShape.particleOrSoftbodyId;

        // PxNodeIndex is a PxU64; the remap table is indexed by the TRIMESH's cache ref.
        const uint2 rigidId = Ptr<uint2>(gMc.shapeToRigidRemapTable)[triCacheRef];

        // The single-level filter (5.12): a filter pair for this rigid and tetrahedron
        // suppresses contact generation between them.  A null array is replaced by the
        // softbody descriptor (llvmpipe speculates the first load, see pxgSlangDefFilter).
        if (findRigidVolumeFilter(
                softBodyRigidVolumeFilters(gMc.softbodies + uint64_t(softbodyId) * PXG_SOFTBODY_STRIDE),
                dfNodeU64(rigidId), triCacheRef, softbodyId, tetrahedronInd))
            continue;

        // ---- cloud A: the tetrahedron, already in WORLD space ---------------------
        const uint4 tetIdx = Ptr<uint4>(softbodies[softbodyId].mTetIndices)[tetrahedronInd];
        Ptr<float4> tetVerts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);

        GjkCloud a;
        for (uint k = 0u; k < GJK_MAX_CLOUD; ++k) a.v[k] = float3(0.0f, 0.0f, 0.0f);
        a.n = 4u;
        a.v[0] = tetVerts[tetIdx.x].xyz;
        a.v[1] = tetVerts[tetIdx.y].xyz;
        a.v[2] = tetVerts[tetIdx.z].xyz;
        a.v[3] = tetVerts[tetIdx.w].xyz;

        float3 tetCenter;
        const float innerSphere = computeInnerSphere(a.v[0], a.v[1], a.v[2], a.v[3], tetCenter);

        // ---- cloud B: the triangle, out of the cooked blob and into WORLD ---------
        // The same blob step 1 walked: header uint4 | BV32DataPacked[header.w] | float4 verts
        // | uint4 indices.  header.w is nbPackedNodes and sizes the node array; header.z is a
        // different count (see softBodyMeshMidphaseBda's header).
        const uint64_t meshPtr = uint64_t(triShape.hullOrMeshOffset.x) |
                                 (uint64_t(triShape.hullOrMeshOffset.y) << 32);
        const uint4 meshHeader = Ptr<uint4>(meshPtr)[0];
        const uint64_t vertsAddr = meshPtr + 16UL + uint64_t(meshHeader.w) * BV32_STRIDE;
        const uint64_t indicesAddr = vertsAddr + uint64_t(meshHeader.x) * 16UL;
        const uint4 triVertInds = Ptr<uint4>(indicesAddr)[triangleInd];
        Ptr<float4> meshVerts = Ptr<float4>(vertsAddr);

        // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
        const uint64_t xf = gMc.transformCache + uint64_t(triCacheRef) * 32UL;
        const float4 triQ = Ptr<float4>(xf)[0];
        const float3 triP = float3(Ptr<float>(xf + 16UL)[0],
                                   Ptr<float>(xf + 20UL)[0],
                                   Ptr<float>(xf + 24UL)[0]);
        const float3 triScale = float3(triShape.scaleX, triShape.scaleY, triShape.scaleZ);
        const float4 triScaleRot = float4(triShape.scaleRotX, triShape.scaleRotY,
                                          triShape.scaleRotZ, triShape.scaleRotW);

        GjkCloud b;
        for (uint k = 0u; k < GJK_MAX_CLOUD; ++k) b.v[k] = float3(0.0f, 0.0f, 0.0f);
        b.n = 3u;
        b.v[0] = mcQuatRotate(triQ, mcVertex2Shape(meshVerts[triVertInds.x].xyz, triScale, triScaleRot)) + triP;
        b.v[1] = mcQuatRotate(triQ, mcVertex2Shape(meshVerts[triVertInds.y].xyz, triScale, triScaleRot)) + triP;
        b.v[2] = mcQuatRotate(triQ, mcVertex2Shape(meshVerts[triVertInds.z].xyz, triScale, triScaleRot)) + triP;

        const float3 triCenter = (b.v[0] + b.v[1] + b.v[2]) * (1.0f / 3.0f);

        // *** THE CROSS PRODUCT IS THE OTHER WAY ROUND. ***  Upstream builds
        // v20 = vB[2] - vB[0], v21 = vB[1] - vB[0], n = normalize(cross(v20, v21)) -- the
        // NEGATIVE of the familiar winding normal.  See the header.
        const float3 v20 = b.v[2] - b.v[0];
        const float3 v21 = b.v[1] - b.v[0];
        const float3 n = mcGetNormalized(cross(v20, v21));

        GjkOut o;
        const uint result = tetPrimitivesCollide2(a, b, tetCenter - triCenter, cDistance,
                                                  innerSphere, TRI_INNER_SPHERE, o);

        bool intersect = (result == GJK_OVERLAP || result == GJK_CLOSE);
        if (intersect)
            intersect = dot(o.direction.xyz, n) >= 0.0f;
        if (!intersect)
            continue;

        // writeVolumeContactFullWarp (softbodyPrimitives.cu, 5.12): the stored point is
        // the witness on the SECOND cloud and the penetration is the witness separation
        // projected on GJK/EPA's direction, normalised (upstream normalises before the call).
        //
        // *** THE STORED NORMAL IS NEGATED. ***  This call site passes `flipOutputNormal = true`
        // (:889) where the hull path passes false (:568), so the xyz that reaches the solver is
        // -direction.  The PENETRATION is computed from the UNFLIPPED direction first and is
        // not negated with it -- the helper computes pen before applying the flip.
        //
        // The clouds are the reason: on this path cloud A is the TETRAHEDRON and cloud B the
        // TRIANGLE, so GJK's direction runs deformable -> rigid, the opposite of the hull
        // path's rigid -> deformable.  The flip is what puts both back on the solver's
        // convention.  Getting it wrong does not lose contacts, it pushes the softbody INTO
        // the floor: measured, the cube accelerated past free fall on first contact.
        const float3 wA = o.closestPointA.xyz;
        const float3 wB = o.closestPointB.xyz;
        const float3 nrm = mcGetNormalized(o.direction.xyz);
        // P9.89: the CUDA sums the three lanes' products and then adds penOffset,
        // pen = ((p0 + p1) + p2) + penOffset (softbodyPrimitives.cu:339), and this call site
        // passes a literal 0 (:888-889).  The association is OpDot's under the NoContraction
        // stamp; the `+ 0` leaf is not a no-op: it turns a -0 sum into +0.  A tet that
        // touches the mesh exactly goes to EPA with lowerBound = +-0, so wB == wA, every
        // lane is +0 * nrm_i, and a normal of (-0, -1, -0) makes the bare dot -0.  Spelled as
        // a select, like the HF sibling's P9.86 fix (softBodyHfContactGenBda.slang:398-402),
        // because a literal `penSum + 0.0f` is folded away (measured: that spelling stores -0
        // again).  testSoftBodySim softBodySimMeshContactGen_p89PenOffset: 24 exact touches,
        // 72 separated contacts against a bitwise mirror of :339.
        const float penSum = dot(wB - wA, nrm);
        const float pen = (penSum == 0.0f) ? 0.0f : penSum;

        uint index;
        InterlockedAdd(Ptr<uint>(gMc.w_totalContactCount)[0], 1u, index);
        if (index >= gMc.w_maxNumContacts)
            continue;

        // 5.12: writeSortKey(index, rigidRef) on every path -- 5.6.1's alternativeWriteMode
        // (contactSortedByX only, the u64 keys untouched) is gone.
        mcWriter().writeVolumeContact(index,
                                      float4(wB.x, wB.y, wB.z, restDistance[cmIdx]),
                                      float4(-nrm.x, -nrm.y, -nrm.z, pen),
                                      dfNodeU64(rigidId), triShape.materialIndex,
                                      PxgEncodeDeformableTetrahedron(softbodyId, tetrahedronInd));
    }
    }

    // The tail reports the UNCLAMPED demand, so the host can grow the stack for the next frame
    // even when this one was truncated.  One thread does it, as CUDA does.
    if (gid.x == 0u)
        InterlockedMax(Ptr<uint>(gMc.stackSizeNeededOnDevice)[0], numFound * 16u);
    }
}
