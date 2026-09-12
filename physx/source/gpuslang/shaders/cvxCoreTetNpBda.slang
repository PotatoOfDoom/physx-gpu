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

// cvxCoreTetNpBda.slang -- `convexCoreTetmeshNphase_Kernel32`
// (convexCoreCollision.cu:1095-1192) on the engine BDA ABI.  P9.72.
//
// ===========================================================================
// THE NAME SAYS SIBLING; THE CODE SAYS OTHERWISE
// ===========================================================================
//
// It sits directly below convexCoreTrimeshNphase_Kernel32 in the same .cu, shares
// its TestBvh instantiation, its _Shared slot, its grid and its traversal -- and it
// is a DIFFERENT KIND OF KERNEL.  The trimesh one is rigid-vs-rigid narrowphase and
// ends in TestOutput::writeOutput.  This one is RIGID-VS-SOFTBODY and ends in
// PxgFEMContactWriter.  Planning it as "the trimesh kernel with tetrahedra" would
// have been wrong in eight separate places:
//
//   NINE parameters, not eighteen, and the last is a STRUCT BY VALUE.  So the
//   136-byte kCvxCorePlane row that the plane, convex and trimesh kernels share is
//   NOT this kernel's row; it needs its own, 144 bytes, ending in an 80-byte writer.
//
//   NO writeOutput.  No contactStream, no patchStream, no cmOutputs, no
//   PxgPatchAndContactCounters.  pxgSlangCvxCoreOut32.slang -- the multi-patch tail
//   P9.70c wrote and P9.71 finally made checkable -- IS NOT USED HERE AT ALL.
//
//   NO Gu::Contact32 accumulator, so pxgSlangCvxCoreAcc.slang is unused too.  There
//   is no patch reduction of any kind: each lane reserves its own slots with one
//   atomicAdd and writes its own contacts.
//
//   generateContacts is the SIX-argument overload (:1174), no cull direction.  The
//   trimesh kernel passes getTriNormal(triIndex) as cullDir and the test is live
//   there; here there is no cullDir and nothing is culled.
//
//   TetmeshBvh::scaleBounds IS THE IDENTITY (:1060-1063) and createConvex takes the
//   vertices RAW (:1075-1079).  A softbody's tet mesh carries no PxMeshScale.
//
//   FOUR vertices per primitive, not three.
//
//   restDistance is a real pointer here (:1146); the trimesh kernel passes NULL.
//
//   It runs on the SOFTBODY stream (PxgNarrowphaseCore.cpp:1051), not mStream.
//
// ===========================================================================
// WHAT IS REUSED, AND IT IS MOST OF THE FILE
// ===========================================================================
//
//   pxgSlangCvxCoreShape (P9.67b)  ShapeRef / PtsRef / supportLocalQ / contactFaceQ
//   pxgSlangCvxCoreCvx   (P9.69a)  gjkDistanceQ / epaDepthQ / faceClipQ
//   pxgSlangCvxCoreMake  (P9.69b)  Gu::makeConvexShape
//   pxgSlangCvxCoreBvh   (P9.70c)  the callback margin, intersectBox, and P9.72's
//                                  cvxBvhCreateConvexTet
//   pxgSlangSoftBodyDesc           the PxgSoftBody mirror
//   pxgSlangBv32         (P8.25)   the BV32DataPacked reader
//
// The contact write is transcribed from softBodyContactGenBda.slang:515-532, which
// is the corpus's canonical BDA spelling of
// PxgFEMContactWriter::writeRigidVsDeformableContactNoBarycentric.
//
// NOT from softBodyMeshContactGenBda.slang:206-221 `writeRigidVsDeformableNoBaryAlt`,
// which has almost the same name and is a DIFFERENT FUNCTION: it is the
// alternativeWriteMode variant, writing contactSortedByX and leaving contactByX and
// tempContactByX untouched.  Lifting that one instead would produce contacts the
// sort never sees, with nothing failing loudly.
//
// ===========================================================================
// THE GRID IS THE TRIMESH KERNEL'S, AND SO IS THE SERIAL PARTITION
// ===========================================================================
//
//     const PxU32 testIndex = threadIdx.y + blockIdx.x * blockDim.y;   // :1113
//
// block (32, 4, 1), grid ceil(numTests/4), one warp per test.  The bridge forwards
// gridDim only and drops blockDim, so [numthreads] IS the block and the four warps
// are walked SERIALLY -- 32 x 4 = 128 = the block, exactly.  tools/check_block_coverage.py
// is what holds this to the arithmetic.

import pxgSlangNpAbi;           // struct mirrors and quaternion helpers
import pxgSlangCvxCoreShape;    // ShapeRef / PtsRef / supportLocalQ / contactFaceQ
import pxgSlangBv32;            // the BV32DataPacked reader
import pxgSlangSoftBodyDesc;    // PxgSoftBodyGpu

#include "pxgSlangCvxCoreCvx.slang"    // gjkDistanceQ / epaDepthQ / faceClipQ
#include "pxgSlangCvxCoreBvh.slang"    // the BVH glue, including the tet arm

// P9.57: FROM THE ENGINE ENUM's numbering, never a remembered literal.
static const uint EGT_SPHERE = 0u;
static const uint EGT_PLANE = 1u;
static const uint EGT_CAPSULE = 2u;
static const uint EGT_BOX = 3u;
static const uint EGT_CONVEXCORE = 4u;
static const uint EGT_CONVEXMESH = 5u;
static const uint EGT_TETRAHEDRONMESH = 7u;   // PxGeometryType::eTETRAHEDRONMESH
// (SPHERE 0, PLANE 1, CAPSULE 2, BOX 3, CONVEXCORE 4, CONVEXMESH 5, PARTICLESYSTEM 6,
//  TETRAHEDRONMESH 7, TRIANGLEMESH 8, HEIGHTFIELD 9.)  I first wrote 9 here, which is
// eHEIGHTFIELD, and check_geom_type_constants.py -- the P9.57 checker, which reads
// PxGeometry.h and compares -- failed the build on it.  The cost of the wrong value
// would have been silent: checkTypes swaps only when shape1 IS the core and shape0 IS
// the mesh, so with 9 the arm never fires and a mesh-first pair is handled with the
// two slots the wrong way round.

#include "pxgSlangCvxCoreMake.slang"   // Gu::makeConvexShape, all five arms

static const uint CVXCORE_TET_WARP = 32u;        // blockDim.x, and the group
static const uint CVXCORE_TET_BLOCK_WARPS = 4u;  // blockDim.y, walked serially
static const uint CVXCORE_TET_STACK = 320u;      // TestBvh::NODE_STACK_MAX
static const uint MAX_CONVEX_CONTACTS = 4u;      // Gu::MAX_CONVEX_CONTACTS
static const uint PXG_BITSHIFT_TET_ID = 20u;     // PxgSimulationCoreDesc.h:40

// ---------------------------------------------------------------------------
// Push constant: the CUDA parameter list EXACTLY (:1095-1104).  Nine arguments,
// the last of them PxgFEMContactWriter BY VALUE -- nine device pointers plus
// PxU32 maxNumContacts plus a 4-byte tail pad, 80 bytes
// (PxgFEMCore.h:361-375; there is a static_assert on the size at
// tests/testFemClothSdfCg.cpp:140).
//
//   numTests 4 @0 (+4 pad), seven pointers @8..@56, the writer at @64 -- already
//   16-aligned, so paramAlign moves nothing -> a 144-byte blob.
// ---------------------------------------------------------------------------
struct CvxCoreTetParams
{
    uint numTests;                    // @0   (+4 pad)
    uint64_t cmInputs;                // @8
    uint64_t shapes;                  // @16
    uint64_t softbodies;              // @24
    uint64_t shapeToRigidRemapTable;  // @32
    uint64_t transformCache;          // @40
    uint64_t contactDistance;         // @48
    uint64_t restDistance;            // @56
    // --- PxgFEMContactWriter, by value (PxgFEMCore.h:361) ---
    uint64_t w_outPoint;              // @64   float4*
    uint64_t w_outNormalPen;          // @72   float4*
    uint64_t w_outBarycentric;        // @80   float4*  (unused on this path)
    uint64_t w_outContactInfo;        // @88   PxgFemOtherContactInfo*  (16 B stride)
    uint64_t w_totalContactCount;     // @96   PxU32*
    uint64_t w_contactByX;            // @104  PxU64*
    uint64_t w_tempContactByX;        // @112  PxU32*
    uint64_t w_contactIndexSortedByX; // @120  PxU32*
    uint64_t w_contactSortedByX;      // @128  PxU32*  (unused on this path)
    uint w_maxNumContacts;            // @136  PxU32
    uint _pad140;                     // @140 -> 144
}

[[vk::push_constant]]
ConstantBuffer<CvxCoreTetParams> gTet;

// ---------------------------------------------------------------------------
// Warp emulation scratch.  One group == one virtual warp; the four tests reuse it
// in turn, separated by the barrier at the top of each trip.
// ---------------------------------------------------------------------------
groupshared int sTetStack[CVXCORE_TET_STACK];
groupshared uint sTetBallot;

uint tetBallot(bool pred, uint lane)
{
    if (lane == 0u) sTetBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sTetBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sTetBallot;
}

// PxEncodeSoftBodyIndex (PxgSimulationCoreDesc.h:110-116).  Same spelling as
// npSbPrim.slang:226 and npSbRemap.slang:146.
uint encodeSoftBody(uint sbId, uint tetId) { return (sbId << PXG_BITSHIFT_TET_ID) | tetId; }

// Gu::ConvexShape::support (GuConvexSupport.h:539-540) -- WITH the margin.  NOT
// pxgSlangCvxCoreCvx's convexBounds, which is Convex<Support>::getBounds: that one
// uses the pose's inverse basis and no margin.
float3 cvxCoreTetSupport(ShapeRef s, PtsRef pts, float margin, float3 dir)
{
    return poseTransform(s, supportLocalQ(s, pts, quatRotInvQ(s.poseQ, dir))) + dir * margin;
}

// Gu::ConvexShape::computeBounds (GuConvexSupport.h:554-561).
void cvxCoreTetComputeBounds(ShapeRef s, PtsRef pts, float margin, out float3 mn, out float3 mx)
{
    const float3 X = float3(1, 0, 0), Y = float3(0, 1, 0), Z = float3(0, 0, 1);
    mn = float3(cvxCoreTetSupport(s, pts, margin, -X).x,
                cvxCoreTetSupport(s, pts, margin, -Y).y,
                cvxCoreTetSupport(s, pts, margin, -Z).z);
    mx = float3(cvxCoreTetSupport(s, pts, margin, X).x,
                cvxCoreTetSupport(s, pts, margin, Y).y,
                cvxCoreTetSupport(s, pts, margin, Z).z);
}

// ===========================================================================
// Gu::generateContacts, the SIX-argument overload (GuConvexSupport.h:1203-1207),
// which forwards PxVec3(0) as the cull direction to the seven-argument one.  The
// cull test `cullDir.dot(axis) < -testEps` is therefore always false here and the
// branch is written out rather than deleted, the same way cvxCoreCvxNpBda does it.
// ===========================================================================
uint cvxCoreTetGenerateContacts(ShapeRef shA, PtsRef ptsA, float marginA,
                                ShapeRef shB, PtsRef ptsB, float marginB,
                                float contactDist,
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

    // cullDir is PxVec3(0) on this overload, so dot(0, axis) < -1e-5f is false.
    const float testEps = 1e-5f;
    if (dot3Q(float3(0, 0, 0), axis) < -testEps)
        return 0u;

    point0 -= axis * marginA;
    point1 += axis * marginB;
    dist -= marginA + marginB;

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

// PxgFEMContactWriter::writeRigidVsDeformableContactNoBarycentric (PxgFEMCore.h:446-465),
// transcribed from softBodyContactGenBda.slang:515-532.  setRigidMaterialIndex(mat)
// followed by markInCollision(false) collapses to a single store of
// (mat & 0x7fffffff).  outBarycentric and contactSortedByX are deliberately NOT
// written on this path, and totalContactCount is the CALLER's to reserve.
void tetWriteRigidVsDeformableNoBary(uint index, float4 contact, float4 normalPen,
                                     uint2 pairInd0, uint pairInd1, uint2 rigidId, uint rigidMatId)
{
    if (index >= gTet.w_maxNumContacts)
        return;
    Ptr<uint2>(gTet.w_contactByX)[index] = rigidId;
    Ptr<uint>(gTet.w_tempContactByX)[index] = rigidId.x;   // PxU32(rigidId & 0xffffffff)
    Ptr<uint>(gTet.w_contactIndexSortedByX)[index] = index;
    Ptr<float4>(gTet.w_outPoint)[index] = contact;
    Ptr<float4>(gTet.w_outNormalPen)[index] = normalPen;
    // PxgFemOtherContactInfo (16 B): pairInd0 u64 @0, pairInd1 u32 @8, rigidMatInd @12.
    const uint64_t ci = gTet.w_outContactInfo + uint64_t(index) * 16UL;
    Ptr<uint2>(ci)[0] = pairInd0;
    Ptr<uint>(ci + 8UL)[0] = pairInd1;
    Ptr<uint>(ci + 12UL)[0] = rigidMatId & 0x7fffffffu;
}

// ===========================================================================
[shader("compute")]
[numthreads(CVXCORE_TET_WARP, 1, 1)]
void convexCoreTetmeshNphase_Kernel32(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    Ptr<PxgShapeGpu> shapesP = Ptr<PxgShapeGpu>(gTet.shapes);
    Ptr<PxsCachedTransformGpu> xfP = Ptr<PxsCachedTransformGpu>(gTet.transformCache);

    for (uint sub = 0u; sub < CVXCORE_TET_BLOCK_WARPS; ++sub)
    {
        const uint testIndex = gid.x * CVXCORE_TET_BLOCK_WARPS + sub;
        GroupMemoryBarrierWithGroupSync();   // the scratch is reused per test
        if (testIndex >= gTet.numTests)
            continue;                        // group-uniform

        // ---- TestInput (:66-96) + checkTypes (:98-108) ----------------------
        const PxgContactManagerInputGpu cin = Ptr<PxgContactManagerInputGpu>(gTet.cmInputs)[testIndex];
        uint cacheRef0 = cin.transformCacheRef0;
        uint cacheRef1 = cin.transformCacheRef1;
        PxgShapeGpu shape0 = shapesP[cin.shapeRef0];
        PxgShapeGpu shape1 = shapesP[cin.shapeRef1];

        // checkTypes(eCONVEXCORE, eTETRAHEDRONMESH).  The swap is live: the `flip` at
        // PxgNphaseImplementationContext.cpp:1602 is local to the bucket selection and
        // registerContactManagerInternal gets the UNFLIPPED workUnit, so a pair can
        // arrive mesh-first.
        if (shape1.type == EGT_CONVEXCORE && shape0.type == EGT_TETRAHEDRONMESH)
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

        const float contactDist = Ptr<float>(gTet.contactDistance)[cacheRef0]
                                + Ptr<float>(gTet.contactDistance)[cacheRef1];
        // restDist = restDistance ? restDistance[testIndex] : 0 (:94).  Unlike the
        // trimesh kernel, this one is HANDED a real pointer, so the live arm is taken.
        const float restDist = Ptr<float>(gTet.restDistance)[testIndex];

        // transform0in1 = transform1.transformInv(transform0) (:1147).
        const float4 conjQ1 = float4(-q1.x, -q1.y, -q1.z, q1.w);
        const float4 aToBq = npAbiQuatMul(conjQ1, q0);
        const float3 aToBp = npAbiQuatRotateInv(q1, p0 - p1);

        ShapeRef convex;
        float convexMargin;
        PtsRef convexPts;
        const bool okConvex = makeConvexShapeCvxCore(shape0, aToBq, aToBp,
                                                     convex, convexMargin, convexPts);

        // ---- convexBounds = convex.computeBounds(); fattenFast(contactDist) ----
        float3 cbMin, cbMax;
        cvxCoreTetComputeBounds(convex, convexPts, convexMargin, cbMin, cbMax);
        cbMin -= float3(contactDist, contactDist, contactDist);
        cbMax += float3(contactDist, contactDist, contactDist);

        // ---- TetmeshBvh(shape1, softbodies) (:1049-1058) ---------------------
        const uint softbodyId = shape1.particleOrSoftbodyId;
        const PxgSoftBodyGpu sb = Ptr<PxgSoftBodyGpu>(gTet.softbodies)[softbodyId];
        // `tetmeshGeomPtr += sizeof(uint4)` -- skip the
        // nbVerts_nbTets_maxDepth_nbBv32TreeNodes header (:1055).
        const uint64_t treeAddr = sb.mTetMeshData + 16UL;
        Ptr<float4> tetVerts = Ptr<float4>(sb.mPosition_InvMass);
        Ptr<uint4> tetIndices = Ptr<uint4>(sb.mTetIndices);

        // scaleBounds is the IDENTITY here (:1060-1063).  The query box goes to the
        // traversal in the frame it was computed in; there is no mesh scale.
        const float3 vMin = cbMin;
        const float3 vMax = cbMax;
        const float tetMargin = cvxBvhCallbackMargin(vMin, vMax);

        // rigidId / softbodyId, read once by the CUDA's lane 0 (:1153-1154).
        // PxNodeIndex is a u64; getInd() is the whole word, and the corpus carries it
        // as a uint2 through the writer.
        const uint64_t rigidIdW = Ptr<uint64_t>(gTet.shapeToRigidRemapTable)[cacheRef0];
        const uint2 rigidId = uint2(uint(rigidIdW & 0xFFFFFFFFUL), uint(rigidIdW >> 32));
        const uint rigidMatId = shape0.materialIndex;

        // ---- bv32TreeTraversal (bv32Traversal.cuh:64-195) --------------------
        uint currentNodeIndex = 0u;
        bool hasNode = okConvex;
        uint nbStack = 0u;

        while (hasNode)
        {
            const uint nodeIdx = bvNodeIdxOf(currentNodeIndex);

            if (bvIsLeaf(currentNodeIndex))
            {
                const uint leafOffset = bvLeafSlotOf(currentNodeIndex);
                const uint data = bvData(treeAddr, nodeIdx, leafOffset);
                // One lane per primitive: CUDA drops past 32 and so does this.
                const uint nbPrimitives = min(bvDataNbPrimitives(data), CVXCORE_TET_WARP);
                const uint primStartIndex = bvDataPrimStart(data);

                uint primitiveIndex = 0xffffffffu;
                if (lane < nbPrimitives)
                    primitiveIndex = lane + primStartIndex;
                // P9.28: the sentinel guard is PER-LANE, so this platform issues the
                // reads for the excluded lanes too.  safePrim is the ADDRESS those
                // reads use; the sentinel still decides the VALUE.
                const uint safePrim = min(primitiveIndex,
                                          primStartIndex + max(nbPrimitives, 1u) - 1u);

                uint numPoints = 0u;
                float3 normal = float3(0, 0, 0);
                float4 patch[MAX_CONVEX_CONTACTS];
                for (uint i = 0u; i < MAX_CONVEX_CONTACTS; ++i) patch[i] = float4(0, 0, 0, 0);

                if (primitiveIndex != 0xffffffffu)
                {
                    const uint4 inds = tetIndices[safePrim];
                    // RAW vertices: no mesh-scale transform anywhere (:1075-1079).
                    const float3 tv0 = tetVerts[inds.x].xyz;
                    const float3 tv1 = tetVerts[inds.y].xyz;
                    const float3 tv2 = tetVerts[inds.z].xyz;
                    const float3 tv3 = tetVerts[inds.w].xyz;

                    ShapeRef tet;
                    PtsRef tetPts;
                    float tetPrimMargin;
                    cvxBvhCreateConvexTet(tv0, tv1, tv2, tv3, tetMargin,
                                          tet, tetPts, tetPrimMargin);

                    numPoints = cvxCoreTetGenerateContacts(convex, convexPts, convexMargin,
                                                           tet, tetPts, tetPrimMargin,
                                                           contactDist, normal, patch);
                }

                // ---- the emit (:1173-1190) ---------------------------------
                // NO warp aggregation: the CUDA's atomicAdd is inside
                // `if (numPoints)`, so every contributing lane reserves its OWN run
                // of slots independently.  Aggregating them into one reservation
                // would change the ORDER contacts land in, which the downstream sort
                // keys off.  Transcribed per-lane.
                if (numPoints != 0u)
                {
                    const uint pairInd1 = encodeSoftBody(softbodyId, primitiveIndex);
                    uint index;
                    InterlockedAdd(Ptr<uint>(gTet.w_totalContactCount)[0], numPoints, index);
                    for (uint i = 0u; i < MAX_CONVEX_CONTACTS; ++i)
                    {
                        if (i >= numPoints) break;
                        // point = points[i] + normal * dists[i] * 0.5f (:1183): the
                        // contact is placed HALFWAY along the penetration, not on
                        // either surface.
                        const float3 point = patch[i].xyz + normal * patch[i].w * 0.5f;
                        // "VR @@@ add fake ERP. it overshoots otherwise" (:1184) --
                        // upstream's own words.  Penetrating distances are scaled to
                        // 40%; separating ones are left alone.  It is a solver hack
                        // and it is part of the observable behaviour.
                        float d = patch[i].w;
                        if (d < 0.0f) d *= 0.4f;
                        const float4 posRest = float4(point, restDist);
                        // THE NORMAL IS NEGATED ON WRITE (:1186).  The FEM side wants
                        // it pointing the other way from what faceClipQ produced.
                        const float4 normDist = float4(-normal.x, -normal.y, -normal.z, d);
                        tetWriteRigidVsDeformableNoBary(index + i, posRest, normDist,
                                                        rigidId, pairInd1, rigidId, rigidMatId);
                    }
                }
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
                    const uint d2 = bvData(treeAddr, nodeIdx, lane);
                    isValidNode = intersect;
                    if (isValidNode && bvDataIsInternal(d2))
                        currentIndex = bvDataChildNodeIndex(d2);
                }

                const uint mask = tetBallot(isValidNode, lane);
                const uint offset = countbits(mask & ((1u << lane) - 1u));
                const uint validCount = countbits(mask);
                if (isValidNode)
                    sTetStack[nbStack + offset] = int(currentIndex);
                nbStack += validCount;
            }

            GroupMemoryBarrierWithGroupSync();
            if (nbStack > 0u)
            {
                nbStack -= 1u;
                currentNodeIndex = uint(sTetStack[nbStack]);
            }
            else
                hasNode = false;
            GroupMemoryBarrierWithGroupSync();
        }
        // There is no writeOutput tail.  The kernel ends with the traversal (:1191).
    }
}
