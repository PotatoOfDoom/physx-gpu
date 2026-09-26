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

// cvxCoreTriNpBda.slang -- `convexCoreTrimeshNphase_Kernel32`
// (convexCoreCollision.cu:918-1040) on the engine BDA ABI.  P9.70c.
//
// The third of the five convexCore*Nphase kernels, and the one that lets a
// PxConvexCoreGeometry rest on a COOKED TRIANGLE MESH.  Until this kernel exists,
// a cylinder dropped on a tessellated mesh floor falls through it: the launch is
// refused in silence and the pair generates no contacts at all.
//
// ===========================================================================
// FIVE LAYERS, ALL OF THEM ALREADY VERIFIED SOMEWHERE ELSE
// ===========================================================================
//
//   pxgSlangCvxCoreShape (P9.67b)  ShapeRef / PtsRef / supportLocalQ /
//                                  contactFaceQ.
//   pxgSlangCvxCoreCvx   (P9.69a)  gjkDistanceQ / epaDepthQ / faceClipQ.
//   pxgSlangCvxCoreMake  (P9.69b)  Gu::makeConvexShape, all five arms.
//   pxgSlangCvxCoreAcc   (P9.70)   Gu::Contact32::addPoints, the full-warp patch
//                                  accumulator -- npCvxCoreWarpAcc verifies every
//                                  output of it BITWISE against a CPU mirror.
//   pxgSlangCvxCoreBvh   (P9.70c)  TrimeshBvh's scaleBounds / margin /
//                                  intersectBox / createConvex / getTriNormal --
//                                  npCvxBvhGlue verifies those bitwise too.
//   pxgSlangBv32         (P8.25)   the BV32DataPacked reader.
//
// What is NEW here is the traversal wiring, the warp emulation, and
// pxgSlangCvxCoreOut32's multi-patch tail -- see that file's header for why the
// tail is a transcription rather than a sixth lift.
//
// ===========================================================================
// THE GRID, AND WHY THE GROUP IS 32 AND NOT 128
// ===========================================================================
//
// CUDA launches grid (ceil(numTests/4), 1, 1) x block (32, 4, 1) and derives
//
//     const PxU32 testIndex = threadIdx.y + blockIdx.x * blockDim.y;   // :945
//
// i.e. ONE WARP PER TEST, four tests per block, each with its OWN `_Shared` slot
// (`__shared__ PxU8 blockShared[BLOCK_WARP_COUNT][sizeof(_Shared)]`, :966).  The
// four warps never interact.  The bridge forwards gridDim only and DROPS blockDim
// (PxgVulkanCudaContext::launchKernel), so the Slang [numthreads] IS the block --
// and this port therefore walks the block's four warps SERIALLY inside a 32-thread
// group, the shape convexTrimeshMidphaseBda and softBodyMeshMidphaseBda both use.
//
//     CUDA           block (32,4,1) = 128 threads, 4 tests, one warp each
//     port           [numthreads(32,1,1)], 4 tests, walked serially
//     grid           ceil(numTests/4), forwarded unchanged
//     coverage       32 x 4 = 128 = the block -- exact
//
// tools/check_block_coverage.py is what holds this to the arithmetic: it is the
// P9.68 checker, written after I shipped a [numthreads(64)] against a block of
// 1024 in P9.67c, and its `serial_factor` condition requires exactly the pairing
// below -- the trip count times the group must cover the block AND the loop
// variable must be added to a group offset scaled by that same bound.
//
// ===========================================================================
// THE SETUP IS COMPUTED BY ALL 32 LANES, NOT BY LANE 0 AND BROADCAST
// ===========================================================================
//
// The .cu writes `_Shared` from lane 0 and __syncwarp()s (:976-989) because
// `TestInput` is 200-odd bytes and one warp's worth of redundant loads is not
// free on a real GPU.  Every input to it is warp-uniform, every operation is
// deterministic, so 32 lanes computing it redundantly reach bit-identical values.
// The port does that instead of staging through groupshared: it is the corpus
// idiom (convexTrimeshMidphaseBda does the same for its per-manager prologue) and
// it removes a barrier from the hot path.
//
// The ACCUMULATOR is the exception and stays on lane 0 alone, because
// cvxCoreAddPoints32 is by construction a one-thread function: it takes all 32
// lanes' contributions as arrays.  Every other lane's copy is dead.
//
// ===========================================================================
// THE STACK IS 320 AND IS NOT CLAMPED
// ===========================================================================
//
// `TestBvh::NODE_STACK_MAX = 320` (:552).  bv32TreeTraversal itself does
// `nbStack += validCount` with no bound (bv32Traversal.cuh:175) -- unlike the
// midphase's own hand-written walk, which clamps.  femClothMeshMidphaseBda, the
// corpus's other port OF THIS SAME FUNCTION, transcribes it unclamped, and so
// does this: inventing a clamp here would silently drop subtrees CUDA visits.
//
// ===========================================================================
// TWO DELIBERATE DIVERGENCES FROM THE .cu, BOTH STATED RATHER THAN HIDDEN
// ===========================================================================
//
// 1. THE QUERY BOUNDS ARE HELD BY VALUE.  `_Callback` stores
//    `const PxBounds3& bounds` bound to the TEMPORARY returned by
//    `bvh.scaleBounds(bounds)` (:566, :592).  Reference lifetime is not extended
//    through a constructor parameter into a member, so that reference dangles for
//    the whole traversal; it works only because the storage is an untouched POD
//    stack slot.  `vMin`/`vMax` below are plain locals.
//
// 2. EMPTY LANES PASS A ZERO NORMAL, NOT AN UNINITIALISED ONE.  The .cu calls
//    `addPoints` OUTSIDE the `if (triIndex != 0xffffffff)` guard (:1007-1009), so a
//    lane with no triangle hands over a `PxVec3 normal` that was declared at :997
//    and never assigned -- and `copyPoints` runs `transform.rotate(normal)` on it
//    unconditionally (:792).  The result is never stored, because the write loop is
//    bounded by that lane's `numPoints == 0`, so the value is unobservable; but
//    reading uninitialised storage is not something to reproduce on purpose, and a
//    NaN through a rotate is not free on every platform.  This port zero-initialises
//    it.  Observationally identical, by the same bound.
//
// THE MARGIN AND THE TRIANGLE ARE IN DIFFERENT UNITS, AND THAT IS UPSTREAM'S.
// `cvxBvhCallbackMargin` is taken from the INVERSE-scaled query bounds (:570-573,
// because the ctor's `_bounds` argument is `bvh.scaleBounds(bounds)`), while
// `createConvex` hands back vertices with the FORWARD scale applied (:636-643) and
// then attaches that same margin to them (:654).  Under a non-unit PxMeshScale the
// two are not the same unit.  Reproduced verbatim; "fixing" it would make this
// kernel disagree with the CPU pipeline.

import pxgSlangNpAbi;           // the struct mirrors and quaternion helpers
import pxgSlangCvxCoreShape;    // ShapeRef / PtsRef / supportLocalQ / contactFaceQ
import pxgSlangBv32;            // the BV32DataPacked reader

#include "pxgSlangCvxCoreCvx.slang"    // gjkDistanceQ / epaDepthQ / faceClipQ
#include "pxgSlangCvxCoreBvh.slang"    // TrimeshBvh's glue
#include "pxgSlangCvxCoreOut.slang"    // the single-patch tail's constants + combiner

// pxgSlangCvxCoreAcc brings its own copies of dot3Q / crossQ / magQ / quatRotQ /
// FLT_MAXQ / NOIDX so that npCvxCoreWarpAcc can include it with no module at all.
// This file already has those from pxgSlangCvxCoreShape, and they are the SAME
// text there (dot3Q, crossQ, quatRotQ character for character; magQ only spelled
// through magSq).  The macro suppresses the duplicates rather than renaming them,
// so the fixture and this kernel still compile one accumulator.
#define PXG_CVXCORE_ACC_HAS_HELPERS 1
#include "pxgSlangCvxCoreAcc.slang"    // Gu::Contact32::addPoints
#include "pxgSlangCvxCoreOut32.slang"  // the multi-patch tail

// P9.57: FROM THE ENGINE ENUM's numbering, never a remembered literal.
static const uint EGT_SPHERE = 0u;
static const uint EGT_PLANE = 1u;
static const uint EGT_CAPSULE = 2u;
static const uint EGT_BOX = 3u;
static const uint EGT_CONVEXCORE = 4u;
static const uint EGT_CONVEXMESH = 5u;
static const uint EGT_TRIANGLEMESH = 8u;

#include "pxgSlangCvxCoreMake.slang"   // Gu::makeConvexShape, all five arms

static const uint CVXCORE_TRI_WARP = 32u;      // blockDim.x, and the group
static const uint CVXCORE_TRI_BLOCK_WARPS = 4u; // blockDim.y, walked serially
static const uint CVXCORE_TRI_STACK = 320u;    // TestBvh::NODE_STACK_MAX

// ---------------------------------------------------------------------------
// Push constant: the CUDA __global__ parameter list EXACTLY (:919-937).  The
// eighteen arguments are byte-identical to convexCorePlaneNphase_Kernel's and
// convexCoreConvexNphase_Kernel's -- verified against the launch site at
// PxgNarrowphaseCore.cpp:963-983 argument by argument -- so all three share the
// 136-byte kCvxCorePlane ABI row rather than carrying three copies of it.
//
// P9.48d is why that row has to be EXACT rather than close: a kernel whose ABI
// row is wrong gets pushConstantSizeFor() == 0 and every launch REFUSED IN
// SILENCE, with fixtures, manifest test, suite and scenes all still green.
// ---------------------------------------------------------------------------
struct CvxCoreTriParams
{
    uint numTests;                    // @0   (+4 pad)
    uint64_t cmInputs;                // @8
    uint64_t cmOutputs;               // @16
    uint64_t shapes;                  // @24
    uint64_t transformCache;          // @32
    uint64_t contactDistance;         // @40
    uint64_t materials;               // @48
    uint64_t contactStream;           // @56
    uint64_t patchStream;             // @64
    uint64_t patchAndContactCounters; // @72
    uint64_t touchChangeFlags;        // @80
    uint64_t patchChangeFlags;        // @88
    uint64_t startContactPatches;     // @96
    uint64_t startContactPoints;      // @104
    uint64_t startContactForces;      // @112
    uint patchBytesLimit;             // @120
    uint contactBytesLimit;           // @124
    uint forceBytesLimit;             // @128 (+4 pad)
    uint _pad132;                     // @132
}                                     // = 136

[[vk::push_constant]]
ConstantBuffer<CvxCoreTriParams> gParams;

// ---------------------------------------------------------------------------
// Warp emulation scratch.  One group == one virtual warp; the four tests reuse it
// in turn, separated by the barrier at the top of each trip.
// ---------------------------------------------------------------------------
groupshared int sStack[CVXCORE_TRI_STACK];
groupshared uint sBallot;

// The staging the accumulator's contract requires: cvxCoreAddPoints32 is called by
// ONE thread with all 32 lanes' contributions already laid out flat, which is the
// shape the CUDA itself produces (copyPoints stages into
// newPoints[WARP_SIZE * MAX_CONVEX_CONTACTS] before the reduction begins).
groupshared uint sLaneCount[WARP_SIZE];
groupshared float4 sLanePts[WARP_SIZE * MAX_CONVEX_CONTACTS];
groupshared float3 sLaneNrm[WARP_SIZE];

uint triBallot(bool pred, uint lane)
{
    if (lane == 0u) sBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sBallot;
}

// Gu::ConvexShape::support (GuConvexSupport.h:539-540) -- WITH the margin.
// NOT the same function as pxgSlangCvxCoreCvx's convexBounds, which is
// Convex<Support>::getBounds: that one uses the pose's inverse basis and NO
// margin.  computeBounds below needs this one, and mixing them up shrinks the
// query box by the core's margin -- which loses contacts only at the edge of the
// contact distance, exactly where a scene is least likely to notice.
float3 cvxCoreShapeSupport(ShapeRef s, PtsRef pts, float margin, float3 dir)
{
    return poseTransform(s, supportLocalQ(s, pts, quatRotInvQ(s.poseQ, dir))) + dir * margin;
}

// Gu::ConvexShape::computeBounds (GuConvexSupport.h:554-561): world-axis supports,
// component-selected.
void cvxCoreComputeBounds(ShapeRef s, PtsRef pts, float margin, out float3 mn, out float3 mx)
{
    const float3 X = float3(1, 0, 0), Y = float3(0, 1, 0), Z = float3(0, 0, 1);
    mn = float3(cvxCoreShapeSupport(s, pts, margin, -X).x,
                cvxCoreShapeSupport(s, pts, margin, -Y).y,
                cvxCoreShapeSupport(s, pts, margin, -Z).z);
    mx = float3(cvxCoreShapeSupport(s, pts, margin, X).x,
                cvxCoreShapeSupport(s, pts, margin, Y).y,
                cvxCoreShapeSupport(s, pts, margin, Z).z);
}

// ===========================================================================
// Gu::generateContacts(ConvexShape, ConvexShape, contactDist, cullDir, ...)
// (GuConvexSupport.h:1169-1201).  The SEVEN-argument overload: the trimesh kernel
// passes the triangle's normal as cullDir (:1005), where the convex-vs-convex
// kernel passes PxVec3(0).  So the cull test below is LIVE here and dead there --
// it is what stops a convex from generating a contact against the BACK of a
// triangle it is passing through.
// ===========================================================================
uint cvxCoreGenerateContacts(ShapeRef shA, PtsRef ptsA, float marginA,
                             ShapeRef shB, PtsRef ptsB, float marginB,
                             float contactDist, float3 cullDir,
                             out float3 normal, out float4 patchOut[MAX_CONVEX_CONTACTS])
{
    normal = float3(0, 0, 0);
    for (uint i = 0u; i < MAX_CONVEX_CONTACTS; ++i) patchOut[i] = float4(0, 0, 0, 0);

    const float maxDist = marginA + marginB + contactDist;

    float dist; uint tagG;
    float3 point0, point1, axis;
    gjkDistanceQ(shA, shB, ptsA, ptsB, maxDist, dist, tagG, point0, point1, axis);

    if (dist < FLT_EPS)
    {
        uint tagE;
        epaDepthQ(shA, shB, ptsA, ptsB, dist, tagE, point0, point1, axis);
    }

    if (dist > maxDist)
        return 0u;

    const float testEps = 1e-5f;
    if (dot3Q(cullDir, axis) < -testEps)
        return 0u;

    point0 -= axis * marginA;
    point1 += axis * marginB;
    dist -= marginA + marginB;

    // generateContactPatch (L872-882): the FaceClipper ctor's two
    // ConvexShape::contactFace wrapper calls, then clip + getContact.
    uint trig0, trig1;
    float3 fn0, fn1;
    float3 fpl[4];
    float3 wfp0[4], wfp1[4];

    const uint np0 = contactFaceQ(shA, ptsA, quatRotInvQ(shA.poseQ, -axis),
                                  quatRotInvQ(shA.poseQ, point0 - shA.poseP), fn0, fpl, trig0);
    const float3 wn0 = quatRotQ(shA.poseQ, fn0);
    for (uint i = 0u; i < 4u; ++i)
        wfp0[i] = i < np0 ? (quatRotQ(shA.poseQ, fpl[i]) + shA.poseP) + wn0 * marginA : float3(0, 0, 0);

    const uint np1 = contactFaceQ(shB, ptsB, quatRotInvQ(shB.poseQ, axis),
                                  quatRotInvQ(shB.poseQ, point1 - shB.poseP), fn1, fpl, trig1);
    const float3 wn1 = quatRotQ(shB.poseQ, fn1);
    for (uint i = 0u; i < 4u; ++i)
        wfp1[i] = i < np1 ? (quatRotQ(shB.poseQ, fpl[i]) + shB.poseP) + wn1 * marginB : float3(0, 0, 0);

    float4 patch[4];
    const uint numPoints = faceClipQ(wfp0, np0, wn0, wfp1, np1, wn1, point0, point1, axis, normal, patch);
    for (uint i = 0u; i < 4u; ++i)
        patchOut[i] = patch[i];
    return numPoints;
}

// ===========================================================================
[shader("compute")]
[numthreads(CVXCORE_TRI_WARP, 1, 1)]
void convexCoreTrimeshNphase_Kernel32(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    Ptr<PxgShapeGpu> shapesP = Ptr<PxgShapeGpu>(gParams.shapes);
    Ptr<PxsCachedTransformGpu> xfP = Ptr<PxsCachedTransformGpu>(gParams.transformCache);

    // The block's four warps, walked serially (see the header on the grid).
    for (uint sub = 0u; sub < CVXCORE_TRI_BLOCK_WARPS; ++sub)
    {
        const uint testIndex = gid.x * CVXCORE_TRI_BLOCK_WARPS + sub;
        GroupMemoryBarrierWithGroupSync();   // the scratch is reused per test
        if (testIndex >= gParams.numTests)
            continue;                        // group-uniform

        // ---- TestInput (:66-96) + checkTypes (:98-108) ----------------------
        const PxgContactManagerInputGpu cin = Ptr<PxgContactManagerInputGpu>(gParams.cmInputs)[testIndex];
        uint cacheRef0 = cin.transformCacheRef0;
        uint cacheRef1 = cin.transformCacheRef1;
        PxgShapeGpu shape0 = shapesP[cin.shapeRef0];
        PxgShapeGpu shape1 = shapesP[cin.shapeRef1];

        // checkTypes(eCONVEXCORE, eTRIANGLEMESH): swap when the pair arrived the
        // other way round.  Unlike the convex-vs-convex kernel -- where either slot
        // may be either thing and there is deliberately NO swap -- this kernel's
        // slot 0 must be the core and slot 1 the mesh, because everything below
        // (makeConvexShape, TrimeshBvh, the material indices) is written that way.
        if (shape1.type == EGT_CONVEXCORE && shape0.type == EGT_TRIANGLEMESH)
        {
            const PxgShapeGpu ts = shape0; shape0 = shape1; shape1 = ts;
            const uint tc = cacheRef0; cacheRef0 = cacheRef1; cacheRef1 = tc;
        }

        const PxsCachedTransformGpu xf0 = xfP[cacheRef0];
        const PxsCachedTransformGpu xf1 = xfP[cacheRef1];
        const float4 q0 = float4(xf0.qx, xf0.qy, xf0.qz, xf0.qw);
        const float3 p0 = float3(xf0.px, xf0.py, xf0.pz);
        const float4 q1 = float4(xf1.qx, xf1.qy, xf1.qz, xf1.qw);
        const float3 p1 = float3(xf1.px, xf1.py, xf1.pz);

        const float contactDist = Ptr<float>(gParams.contactDistance)[cacheRef0]
                                + Ptr<float>(gParams.contactDistance)[cacheRef1];

        // transform0in1 = transform1.transformInv(transform0) (:980).  NOT the
        // pair-midpoint recentring the convex-vs-convex kernel does: there the two
        // convexes are posed about their shared midpoint, here the whole problem is
        // moved into the MESH's frame, which is the frame the BVH's boxes are in.
        // PxTransform::transformInv (PxTransform.h:167-168): qinv = getConjugate();
        // q = qinv * src.q, p = qinv.rotate(src.p - p).  P9.86: rotate by the
        // CONJUGATE, not rotateInv(q1): the two agree on every non-zero value but not
        // on the sign of a zero component (e.g. q1 = PxQuat(-PxHalfPi, X), v = (0,1,-2):
        // CUDA +0, rotateInv -0).  CUDA twin: convexCoreCollision.cu:980.
        const float4 conjQ1 = float4(-q1.x, -q1.y, -q1.z, q1.w);
        const float4 aToBq = npAbiQuatMul(conjQ1, q0);
        const float3 aToBp = npAbiQuatRotate(conjQ1, p0 - p1);

        ShapeRef convex;
        float convexMargin;
        PtsRef convexPts;
        const bool okConvex = makeConvexShapeCvxCore(shape0, aToBq, aToBp,
                                                     convex, convexMargin, convexPts);

        // ---- convexBounds = convex.computeBounds(); fattenFast(contactDist) ----
        float3 cbMin, cbMax;
        cvxCoreComputeBounds(convex, convexPts, convexMargin, cbMin, cbMax);
        cbMin -= float3(contactDist, contactDist, contactDist);
        cbMax += float3(contactDist, contactDist, contactDist);

        // ---- TrimeshBvh(shape1) = readTriangleMesh (triangleMesh.cuh:83-92) ----
        const uint64_t meshPtr = shape1.hullOrMeshPtr;
        const uint4 meshHeader = Ptr<uint4>(meshPtr)[0];
        const uint64_t treeAddr = meshPtr + 16UL;
        // .w is nbPackedNodes and sizes the node ARRAY; .x is the vertex count.
        const uint64_t vertsAddr = treeAddr + uint64_t(meshHeader.w) * BV32_STRIDE;
        const uint64_t indicesAddr = vertsAddr + uint64_t(meshHeader.x) * 16UL;
        Ptr<float4> triVerts = Ptr<float4>(vertsAddr);
        Ptr<uint4> triIndices = Ptr<uint4>(indicesAddr);

        const float3 meshScale = float3(shape1.scaleX, shape1.scaleY, shape1.scaleZ);
        const float4 meshScaleRot = float4(shape1.scaleRotX, shape1.scaleRotY,
                                           shape1.scaleRotZ, shape1.scaleRotW);

        // ---- the traversal's query box, in VERTEX space (:592) ----------------
        float3 vMin, vMax;
        cvxBvhScaleBounds(meshScale, meshScaleRot, cbMin, cbMax, vMin, vMax);
        const float triMargin = cvxBvhCallbackMargin(vMin, vMax);

        // ---- Gu::Contact32() (:985), on lane 0 alone -------------------------
        uint mNumPatches = 0u;
        uint mNumPoints = 0u;
        float3 patchNormal[MAX_PATCHES];
        uint patchCount[MAX_PATCHES];
        float4 mPoints[MAX_PATCHES * MAX_PATCH_POINTS];
        for (uint i = 0u; i < MAX_PATCHES; ++i)
        {
            patchNormal[i] = float3(0, 0, 0);
            patchCount[i] = 0u;
        }
        for (uint i = 0u; i < MAX_PATCHES * MAX_PATCH_POINTS; ++i)
            mPoints[i] = float4(0, 0, 0, 0);

        // ---- bv32TreeTraversal (bv32Traversal.cuh:64-195) --------------------
        uint currentNodeIndex = 0u;
        bool hasNode = okConvex;   // a shape makeConvexShape refuses has no query
        uint nbStack = 0u;

        while (hasNode)
        {
            const uint nodeIdx = bvNodeIdxOf(currentNodeIndex);

            if (bvIsLeaf(currentNodeIndex))
            {
                const uint leafOffset = bvLeafSlotOf(currentNodeIndex);
                const uint data = bvData(treeAddr, nodeIdx, leafOffset);
                // Six-bit field, but ONE LANE PER PRIMITIVE: CUDA maps idxInWarp and
                // silently drops anything past 32.  Transcribed bug for bug -- see
                // pxgSlangBv32's header.
                const uint nbPrimitives = min(bvDataNbPrimitives(data), WARP_SIZE);
                const uint primStartIndex = bvDataPrimStart(data);

                uint primitiveIndex = 0xffffffffu;
                if (lane < nbPrimitives)
                    primitiveIndex = lane + primStartIndex;
                // P9.28: the sentinel guard below is PER-LANE, so this platform turns
                // it into a mask and issues the reads for the EXCLUDED lanes too --
                // at primStartIndex + 0xffffffff, sixty-four gigabytes out.
                // `safePrim` is the ADDRESS those reads use; the sentinel still
                // decides the VALUE, so nothing observable changes.
                const uint safePrim = min(primitiveIndex,
                                          primStartIndex + max(nbPrimitives, 1u) - 1u);

                // _Callback::intersectPrimitiveFullWarp (:583-590) then
                // Callback::operator() (:995-1026).
                uint numPoints = 0u;
                float3 normal = float3(0, 0, 0);
                float4 patch[MAX_CONVEX_CONTACTS];
                for (uint i = 0u; i < MAX_CONVEX_CONTACTS; ++i) patch[i] = float4(0, 0, 0, 0);

                if (primitiveIndex != 0xffffffffu)
                {
                    const uint4 inds = triIndices[safePrim];
                    const float3 rv0 = triVerts[inds.x].xyz;
                    const float3 rv1 = triVerts[inds.y].xyz;
                    const float3 rv2 = triVerts[inds.z].xyz;

                    ShapeRef tri;
                    PtsRef triPts;
                    float triPrimMargin;
                    cvxBvhCreateConvex(meshScale, meshScaleRot, rv0, rv1, rv2, triMargin,
                                       tri, triPts, triPrimMargin);

                    const float3 triNormal = cvxBvhTriNormal(meshScale, meshScaleRot, rv0, rv1, rv2);

                    numPoints = cvxCoreGenerateContacts(convex, convexPts, convexMargin,
                                                        tri, triPts, triPrimMargin,
                                                        contactDist, triNormal, normal, patch);
                }

                // Stage this lane's contribution for the accumulator.  The CUDA calls
                // addPoints for EVERY leaf including one where no lane found anything;
                // with totalPoints == 0 the round loop does not execute, so calling it
                // unconditionally is both faithful and free.
                sLaneCount[lane] = numPoints;
                sLaneNrm[lane] = normal;
                for (uint i = 0u; i < MAX_CONVEX_CONTACTS; ++i)
                    sLanePts[lane * MAX_CONVEX_CONTACTS + i] = patch[i];
                GroupMemoryBarrierWithGroupSync();

                if (lane == 0u)
                {
                    uint laneCount[WARP_SIZE];
                    float4 lanePts[WARP_SIZE * MAX_CONVEX_CONTACTS];
                    float3 laneNrm[WARP_SIZE];
                    for (uint l = 0u; l < WARP_SIZE; ++l)
                    {
                        laneCount[l] = sLaneCount[l];
                        laneNrm[l] = sLaneNrm[l];
                        for (uint j = 0u; j < MAX_CONVEX_CONTACTS; ++j)
                            lanePts[l * MAX_CONVEX_CONTACTS + j] = sLanePts[l * MAX_CONVEX_CONTACTS + j];
                    }
                    // contact.addPoints(normal, points, dists, numPoints, transform1)
                    // (:1009): the points arrive in MESH space and the accumulator is
                    // what rotates them to WORLD.
                    cvxCoreAddPoints32(mNumPatches, mNumPoints, patchNormal, patchCount, mPoints,
                                       laneCount, lanePts, laneNrm, q1, p1);
                }
                GroupMemoryBarrierWithGroupSync();
            }
            else
            {
                const uint nbChildren = bvNbNodes(treeAddr, nodeIdx);
                bool intersect = false;
                if (lane < nbChildren)
                {
                    const float3 mn = bvMin(treeAddr, nodeIdx, lane);
                    const float3 mx = bvMax(treeAddr, nodeIdx, lane);
                    intersect = cvxBvhIntersectBox(mn, mx, vMin, vMax);
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

                const uint mask = triBallot(isValidNode, lane);
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
            GroupMemoryBarrierWithGroupSync();
        }

        // ---- TestOutput::writeOutput (:1033-1039), lane 0 only ---------------
        if (lane == 0u)
        {
            CvxCoreContact32 c;
            c.numPatches = mNumPatches;
            c.numPoints = mNumPoints;
            for (uint i = 0u; i < MAX_PATCHES; ++i)
            {
                c.patchNormal[i] = patchNormal[i];
                c.patchCount[i] = patchCount[i];
            }
            for (uint i = 0u; i < MAX_PATCHES * MAX_PATCH_POINTS; ++i)
                c.points[i] = mPoints[i];

            CvxCoreOutRefs orefs;
            orefs.cmOutputs = gParams.cmOutputs;
            orefs.materials = gParams.materials;
            orefs.contactStream = gParams.contactStream;
            orefs.patchStream = gParams.patchStream;
            orefs.patchAndContactCounters = gParams.patchAndContactCounters;
            orefs.touchChangeFlags = gParams.touchChangeFlags;
            orefs.patchChangeFlags = gParams.patchChangeFlags;
            orefs.startContactPatches = gParams.startContactPatches;
            orefs.startContactPoints = gParams.startContactPoints;
            orefs.startContactForces = gParams.startContactForces;
            orefs.patchBytesLimit = gParams.patchBytesLimit;
            orefs.contactBytesLimit = gParams.contactBytesLimit;
            orefs.forceBytesLimit = gParams.forceBytesLimit;

            cvxCoreWriteOutput32(testIndex, shape0, shape1, c, orefs);
        }
    }
}
