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
// femClothMeshMidphaseBda — cloth_meshMidphaseGeneratePairsLaunch
// (gpunarrowphase/src/CUDA/femClothClothMidPhase.cu:462, core femClothTriMeshMidphaseCore
// at :300, launch PxgNarrowphaseCore.cpp:6579 inside testSDKFemClothTrimesh,
// CLOTH_MESH_MIDPHASE), on the engine BDA ABI.
//
// P8.17 step 1 of TWO for cloth-vs-COOKED-TRIANGLE-MESH — the pair that lets a cloth drape over
// cooked static geometry instead of over a primitive.  One warp per CLOTH TRIANGLE walks the
// mesh's BV32 tree and pushes its four closest mesh triangles onto the shared midphase stack;
// step 2 (cloth_meshContactGenLaunch) turns each pair into contacts.  Both launches are
// unconditional in `testSDKFemClothTrimesh` — there is no `if (0)` guard and no dead-consumer
// problem, unlike the four cloth-VERTEX-vs-mesh/HF kernels P8.14 ruled dead.
//
// THIS IS softBodyMeshMidphaseBda's SIBLING, one step over.  Same cooked blob, same top-4
// extraction, same grid.  The query is a cloth TRIANGLE instead of a softbody TETRAHEDRON, which
// changes four things:
//
//   1. THE SAT IS TRIANGLE-vs-TRIANGLE (`Gu::satIntersect<true>`, triangletriangle.cuh:74), not
//      tet-vs-triangle.  Two plane terms — the query triangle's plane against the found
//      triangle's verts and the reverse — and then the NINE edge-cross axes, each normalized and
//      run through `testSeparatingAxis`.  The tet version has four plane terms and no cross
//      axes.  `TDoCross` is `true` at this call site, so the cross block is live;
//   2. NO `isValidTet` GATE and no surface-hint lookup.  Every cloth triangle is a query — a
//      cloth has no interior primitives to skip, so the softbody path's `mTetMeshSurfaceHint`
//      byte read has no counterpart here;
//   3. THE BOX TEST IS A PLAIN AABB OVERLAP against a PRE-FATTENED query bound
//      (`triangleBound.fattenFast(contactDist)` at :407, then
//      `!(min.x > tMax.x || tMin.x > max.x || ...)` at :174).  The softbody traverser instead
//      keeps an unfattened bound and compares a per-axis DISPLACEMENT against `contactDist^2`.
//      Same name, two different structs in two different .cu files — do not carry one over;
//   4. THE CONTACT DISTANCE IS NOT DIVIDED BY THE SCALE.  `softbodySoftbodyMidPhase.cu` divides
//      the world distance by `scale.abs().minElement()` because it compares against vertex-space
//      separations; `femClothClothMidPhase.cu:358` does NOT, and fattens the vertex-space bound
//      by the raw world distance.  Transcribed as written, not corrected — a "fix" here would
//      change the emitted pair set on any scaled mesh.
//
// The query triangle still goes world -> mesh local -> mesh VERTEX space, both steps
// (`shape2Vertex` DIVIDES by the scale; skipping it works only at scale (1,1,1)), and the cooked
// blob is still walked with `.w` (nbPackedNodes) sizing the node array, NOT `.z`.
//
// WHICH SHAPE IS THE CLOTH.  Both halves of this pair are `eTRIANGLEMESH`, so the type cannot
// disambiguate them the way it does for the softbody pair.  The engine uses
// `particleOrSoftbodyId != 0xffffffff` — a deformable shape carries its body id there and a
// static cooked mesh carries 0xffffffff (:330-343).
//
// *** THE GRID.  READ THIS BEFORE CHANGING numthreads. ***
// The engine launches grid (SB_SBMIDPHASE = 1024, numTests, 1) with block
// (WARP_SIZE, MIDPHASE_WARPS_PER_BLOCK = 2, 1) and grid-strides over the cloth's triangles:
//
//     const PxU32 globalWarpIdx = threadIdx.y + blockIdx.x * blockDim.y;   // [0, 2048)
//     const PxU32 NbWarps       = blockDim.y * gridDim.x;                  // 2048
//
// The bridge forwards gridDimX/Y/Z but DROPS blockDim, so a [numthreads(32,1,1)] group is ONE
// warp: cmIdx = gid.y, the block's two warps are walked serially in a group-uniform `sub` loop,
// and the stride stays the CUDA 2048 rather than the group count.
//
// DEVIATIONS from CUDA (recipe policy, each preserving the emitted pair SET and the atomic
// total): no 32-lane intrinsics (__ballot_sync -> groupshared InterlockedOr, minIndex -> a
// groupshared float[32] plus a strict-`<` scan from lane 0, which picks the same lowest-lane
// minimum); the per-warp femMidphaseScratch is dropped and every lane reads the addresses
// directly through Ptr<>.
//
// The `mIsMesh` branch of the traverser is the one taken here (:135), so the filter-pair lookup
// on the other branch is NOT transcribed — this call site never calls `setFiltering`, so
// `mFilterPairs` is NULL and `mNumFilterPairs` 0 for the whole traversal.
//
// DUPLICATION, DELIBERATELY: the BV32 readers, the quaternion helpers and `shape2Vertex` are a
// third copy (softBodyClothMidphaseBda and softBodyMeshMidphaseBda carry the other two).  P8.15
// extracted a shared module the moment a SECOND consumer appeared, and that is the rule being
// followed here too — `cloth_meshContactGenLaunch` (step 2) is the second consumer of THESE
// helpers, and the extraction belongs in that step where it can be proved behaviour-preserving,
// not folded into a port that is also adding a new kernel.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): every plane product in makePlane / minProject,
// the SAT's dot-of-cross axes and the projection intervals are mul-feeding-add chains that would
// otherwise fuse to FMAs and break parity with the -ffp-contract=off CPU reference.
// ===========================================================================
module femClothMeshMidphaseBda;

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu (mTriMeshData@0, mPosition_InvMass@16, ...)

// PxgNpKernelIndices.h:32 / PxgSoftBodyCoreKernelIndices.h:55.
static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
static const uint SB_SBMIDPHASE = 1024u;
static const uint SM_NB_WARPS = SB_SBMIDPHASE * MIDPHASE_WARPS_PER_BLOCK;

static const uint WARP = 32u;
static const float PX_MAX_F32 = 3.4028235e38f;

// Gu::BV32DataPacked: 1168 B — mMin[32] @0, mMax[32] @512, mData[32] @1024, mNbNodes @1152.
import pxgSlangBv32;   // P8.25: the BV32 reader, shared

// femMidphaseScratch.sBv32Nodes[192] — a 6-deep BV32 tree's worst-case pending set.
static const uint STACK_SIZE = 192u;

// `const PxU32 MaxNumPairs = 4` (femClothClothMidPhase.cu:424).
static const uint MAX_NUM_PAIRS = 4u;

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA pack (PxgNarrowphaseCore.cpp:6581):
// { PxReal toleranceLength, cmInputs*, transformCache*, bounds*, contactDistance*, gpuShapes*,
//   femClothes*, PxU32 stackSizeBytes, stackPtr*, midphasePairsNum* }.
// Natural alignment: 4 @0, six pointers @8..@48, PxU32 @56, then a pointer realigned to @64 and
// @72 -> 80-byte blob.  ABI row {4,8,8,8,8,8,8,4,8,8} — the same SHAPE as
// sb_meshMidphaseGeneratePairsLaunch's, with femClothes where softbodies was.
// ---------------------------------------------------------------------------
struct ClothMeshMidphaseParams
{
    float toleranceLength;          // @0   (ABI parity only — the core never reads it)
    uint _pad4;                     // @4
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16
    uint64_t bounds;                // @24  (ABI parity only)
    uint64_t contactDistance;       // @32
    uint64_t gpuShapes;             // @40
    uint64_t femClothes;            // @48
    uint stackSizeBytes;            // @56
    uint _pad60;                    // @60
    uint64_t stackPtr;              // @64
    uint64_t midphasePairsNum;      // @72 -> 80
}

[[vk::push_constant]] ConstantBuffer<ClothMeshMidphaseParams> gCm;

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

// ---------------------------------------------------------------------------
// Geometry.
// ---------------------------------------------------------------------------
float3 cmGetNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

// PxPlane(p0, p1, p2): n = normalize(cross(p1-p0, p2-p0)), d = -p0.dot(n).
void makePlane(float3 p0, float3 p1, float3 p2, out float3 n, out float d)
{
    n = cmGetNormalized(cross(p1 - p0, p2 - p0));
    d = -dot(p0, n);
}

// minProject(plane, triangle) (triangletriangle.cuh:43): the `+ plane.d` lands ONCE, AFTER the
// min over the three dots — transcribed in that order rather than folded into each term.
float minProjectTri(float3 n, float d, float3 v0, float3 v1, float3 v2)
{
    return min(dot(n, v0), min(dot(n, v1), dot(n, v2))) + d;
}

// testSeparatingAxis (triangletriangle.cuh:56): the max gap of the two projection intervals.
float testSeparatingAxis(float3 axis, float3 a0, float3 a1, float3 a2,
                         float3 b0, float3 b1, float3 b2)
{
    float min0, max0, min1, max1;
    min0 = max0 = dot(a0, axis);
    min1 = max1 = dot(b0, axis);
    const float pa1 = dot(a1, axis), pb1 = dot(b1, axis);
    min0 = min(pa1, min0); max0 = max(pa1, max0);
    min1 = min(pb1, min1); max1 = max(pb1, max1);
    const float pa2 = dot(a2, axis), pb2 = dot(b2, axis);
    min0 = min(pa2, min0); max0 = max(pa2, max0);
    min1 = min(pb2, min1); max1 = max(pb2, max1);
    return max(min1 - max0, min0 - max1);
}

// Gu::satIntersect<true>(triangle0, triangle1, tolerance) (triangletriangle.cuh:74).  triangle0
// is the QUERY (the cloth triangle), triangle1 the found mesh triangle.  Every early-out is
// `sep > tolerance`, and the caller passes min(bestDistance, contactDistance) so the tolerance
// TIGHTENS as the traversal finds closer triangles.
float satIntersectTriTri(float3 a0, float3 a1, float3 a2, float3 nA, float dA,
                         float3 b0, float3 b1, float3 b2, float3 nB, float dB,
                         float tolerance)
{
    float sep = minProjectTri(nA, dA, b0, b1, b2);
    if (sep > tolerance) return sep;

    sep = max(sep, minProjectTri(nB, dB, a0, a1, a2));
    if (sep > tolerance) return sep;

    const float3 axes0[3] = { a1 - a0, a2 - a0, a2 - a1 };
    const float3 axes1[3] = { b1 - b0, b2 - b0, b2 - b1 };

    for (uint i = 0u; i < 3u; ++i)
    {
        const float3 axis0 = axes0[i];
        for (uint j = 0u; j < 3u; ++j)
        {
            float3 sepAxis = cross(axis0, axes1[j]);
            const float magSq = dot(sepAxis, sepAxis);
            if (magSq > 1e-5f)
            {
                sepAxis = sepAxis * (1.0f / sqrt(magSq));
                sep = max(sep, testSeparatingAxis(sepAxis, a0, a1, a2, b0, b1, b2));
                if (sep > tolerance) return sep;
            }
        }
    }
    return sep;
}

// PxQuat::rotate / rotateInv (PxQuat.h:310, :324), literal.
float3 cmQuatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}
float3 cmQuatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// shape2Vertex(v, scale, rot) = rot.rotateInv(rot.rotate(v) / scale)  (cudaNpCommon.h:285).
// DIVIDES by the scale — it is the inverse of vertex2Shape, which multiplies.
float3 cmShape2Vertex(float3 v, float3 scale, float4 rot)
{
    float3 temp = cmQuatRotate(rot, v);
    temp.x /= scale.x;
    temp.y /= scale.y;
    temp.z /= scale.z;
    return cmQuatRotateInv(rot, temp);
}

// femClothTriMeshMidphaseCore:320-344.  BOTH shapes are eTRIANGLEMESH here, so the CLOTH is the
// one carrying a body id; the cooked static mesh carries 0xffffffff.
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
groupshared int sCmStack[STACK_SIZE];   // bv32 to-be-expanded node stack
groupshared uint sCmBallot;             // __ballot_sync mask
groupshared float sCmBest[WARP];        // minIndex reduction over the lane bests
groupshared uint sCmIndex0;             // atomicAdd result broadcast

uint cmBallot(bool pred, uint lane)
{
    if (lane == 0u) sCmBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sCmBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sCmBallot;
}

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_meshMidphaseGeneratePairsLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint cmIdx = gid.y;                      // blockIdx.y — one pair per y-block

    PxgShape clothShape, triShape;
    uint clothCacheRef, triCacheRef;
    loadClothTrimeshPair(gCm.cmInputs, gCm.gpuShapes, cmIdx,
                         clothShape, clothCacheRef, triShape, triCacheRef);

    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gCm.femClothes);
    const uint clothId = clothShape.particleOrSoftbodyId;

    // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
    const uint64_t xf = gCm.transformCache + uint64_t(triCacheRef) * 32UL;
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
    Ptr<float4> meshVerts = Ptr<float4>(vertsAddr);
    Ptr<uint4> meshIndices = Ptr<uint4>(indicesAddr);

    // femClothClothMidPhase.cu:358 — the raw SUM, NOT divided by the scale.  See the header,
    // difference 4.
    const float contactDist = Ptr<float>(gCm.contactDistance)[clothCacheRef] +
                              Ptr<float>(gCm.contactDistance)[triCacheRef];

    const uint nbTriangles = clothes[clothId].mNbTriangles;
    Ptr<uint4> clothTriIndices = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices);
    Ptr<float4> clothVerts = Ptr<float4>(clothes[clothId].mPosition_InvMass);

    const uint stackCapacity = gCm.stackSizeBytes / 16u;

    // The block's two warps, walked serially (see the header on the grid).
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    const uint globalWarpIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;

    for (uint triangleIdx = globalWarpIdx; triangleIdx < nbTriangles; triangleIdx += SM_NB_WARPS)
    {
        GroupMemoryBarrierWithGroupSync();   // groupshared scratch is reused per query

        // P9.49: safe ADDRESS for the read this platform issues even when the loop
        // condition admits no lane (see femClothPsMidphaseBda, where it faulted).
        // The three vertex reads below need no clamp of their own: for every index
        // this one can now yield, the triangle it returns is a real one.
        const uint4 vertInds = clothTriIndices[min(triangleIdx, max(nbTriangles, 1u) - 1u)];

        // World -> mesh local -> mesh VERTEX space, all three corners.
        const float3 a0 = cmShape2Vertex(cmQuatRotateInv(triQ, clothVerts[vertInds.x].xyz - triP),
                                         triScale, triScaleRot);
        const float3 a1 = cmShape2Vertex(cmQuatRotateInv(triQ, clothVerts[vertInds.y].xyz - triP),
                                         triScale, triScaleRot);
        const float3 a2 = cmShape2Vertex(cmQuatRotateInv(triQ, clothVerts[vertInds.z].xyz - triP),
                                         triScale, triScaleRot);

        // PxBounds3 over the three corners, then fattenFast(contactDist).  The box test below is
        // a plain overlap against THIS bound — the fattening is the whole tolerance.
        const float3 tMin = min(min(a0, a1), a2) - contactDist;
        const float3 tMax = max(max(a0, a1), a2) + contactDist;

        // Triangle triangle: verts, triPlane = PxPlane(v0,v1,v2), centroid = (v0+v1+v2)/3.
        float3 nA; float dA;
        makePlane(a0, a1, a2, nA, dA);
        // P9.36: femClothClothMidPhase.cu:417/:578 write `(v0 + v1 + v2) / 3.f` into a
        // PxVec3 centroid (triangletriangle.cuh:39), i.e. PxVec3::operator/ -- ONE
        // reciprocal and three multiplies.  1.0f/3.0f is inexact, so this is not a
        // no-op rewrite; the engine spells the same thing `* third` at
        // sdfCollision.cuh:665.
        const float3 centroid = (a0 + a1 + a2) * (1.0f / 3.0f);

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

                if (primitiveIndex != 0xffffffffu)
                {
                    const uint4 mTri = meshIndices[safePrim];
                    const float3 b0 = meshVerts[mTri.x].xyz;
                    const float3 b1 = meshVerts[mTri.y].xyz;
                    const float3 b2 = meshVerts[mTri.z].xyz;

                    float3 nB; float dB;
                    makePlane(b0, b1, b2, nB, dB);

                    // "get rid of double-sided triangle" (:138): the FOUND triangle's plane
                    // against the QUERY's centroid.
                    if (dot(nB, centroid) + dB >= 0.0f)
                    {
                        const float separation =
                            satIntersectTriTri(a0, a1, a2, nA, dA, b0, b1, b2, nB, dB,
                                               min(bestDist, contactDist));
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
                // intersectBoxFullWarp (:171): a plain AABB overlap — the query bound is already
                // fattened.  NOT the softbody traverser's displacement form.
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
                    intersect = !(mn.x > tMax.x || tMin.x > mx.x ||
                                  mn.y > tMax.y || tMin.y > mx.y ||
                                  mn.z > tMax.z || tMin.z > mx.z);

                uint currentIndex = bvChildIndex(currentNodeIndex, lane);
                bool isValidNode = false;
                if (lane < nbChildren)
                {
                    const uint data = bvData(treeAddr, nodeIdx, lane);
                    isValidNode = intersect;
                    if (isValidNode && bvDataIsInternal(data))
                        currentIndex = bvDataChildNodeIndex(data);
                }

                const uint mask = cmBallot(isValidNode, lane);
                const uint offset = countbits(mask & ((1u << lane) - 1u));
                const uint validCount = countbits(mask);
                if (isValidNode)
                    sCmStack[nbStack + offset] = int(currentIndex);
                nbStack += validCount;
            }

            GroupMemoryBarrierWithGroupSync();
            if (nbStack > 0u)
            {
                nbStack -= 1u;
                currentNodeIndex = uint(sCmStack[nbStack]);
            }
            else
                hasNode = false;
        }

        // --- the top-4 extraction (femClothClothMidPhase.cu:424-455) ---------------
        GroupMemoryBarrierWithGroupSync();
        sCmBest[lane] = bestDist;
        const uint mask = cmBallot(bestDist < contactDist, lane);
        uint count = countbits(mask);
        if (count > MAX_NUM_PAIRS) count = MAX_NUM_PAIRS;

        if (count == 0u)
            continue;

        // ONE atomicAdd of the whole count, broadcast to the warp.
        if (lane == 0u)
        {
            uint idx;
            InterlockedAdd(Ptr<uint>(gCm.midphasePairsNum)[0], count, idx);
            sCmIndex0 = idx;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint index = sCmIndex0;

        float bestSep = -1.0f;
        for (uint i = 0u; i < count && bestSep < contactDist; ++i)
        {
            // minIndex: the warp argmin, lowest lane winning ties.
            uint winnerLane = 0u;
            float best = PX_MAX_F32;
            for (uint j = 0u; j < WARP; ++j)
                if (sCmBest[j] < best) { best = sCmBest[j]; winnerLane = j; }
            bestSep = best;

            GroupMemoryBarrierWithGroupSync();
            if (winnerLane == lane && sCmBest[lane] < contactDist)
            {
                if ((index + i) < stackCapacity)
                    Ptr<uint4>(gCm.stackPtr)[index + i] =
                        uint4(cmIdx, triangleIdx, bestPrimIndex, 0u);
                sCmBest[lane] = PX_MAX_F32;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    }
}
