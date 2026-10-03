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

// cvxCoreClothNpBda.slang -- `convexCoreClothmeshNphase_Kernel32`
// (convexCoreCollision.cu:1195-1279) on the engine BDA ABI.  P9.74.
//
// ===========================================================================
// THE FIFTH convexCore*Nphase KERNEL IS THE THIRD DISTINCT SHAPE
// ===========================================================================
//
// The five share a name prefix and almost nothing else, and this one is the
// clearest case yet.  Plane, convex and trimesh end in TestOutput::writeOutput and
// share a 136-byte ABI row.  The tetmesh one (P9.72) ends in PxgFEMContactWriter
// and needed its own 144-byte row.  This one ends in PxgFEMContactWriter TOO -- and
// is still not the tetmesh kernel with triangles, because:
//
//   IT DOES NOT TRAVERSE A BVH AT ALL.  Every other kernel in the family builds a
//   query box and walks bv32TreeTraversal.  This one is a MIDPHASE-PAIR CONSUMER:
//   cloth_midphaseGeneratePairsLaunch has already run on the same stream
//   (PxgNarrowphaseCore.cpp:1143) and left (cmIndex, triIndex) pairs on the stack.
//   This kernel reads them.  There is no node stack, no ballot, no groupshared
//   anything, and no cross-lane communication of any kind -- it is the only member
//   of the family that is purely per-thread.  TrimeshBvh is still constructed, but
//   only for getTriIndices / createConvex / getTriBarycentric; getNodes and
//   scaleBounds are never called.
//
//   THE GRID IS FIXED AND FLAT.  32 blocks x 512 threads, both literals at the
//   launch site (PxgNarrowphaseCore.cpp:1186-1187), with a grid-stride loop
//   `pairIndex += gridDim.x * blockDim.x` over `numPairs`.  Every other kernel here
//   sizes its grid to the test count.
//
//   FOURTEEN parameters, ending in the by-value writer -- a 176-byte blob that
//   matches no existing row.  kCvxCorePlane (136) and kCvxCoreTet (144) are both
//   wrong, and P9.48d is why "close" is fatal rather than approximate: a missing or
//   mismatched row gives pushConstantSizeFor() == 0 and every launch is REFUSED IN
//   SILENCE, with fixtures, manifest test, suite and scenes all still green.
//
//   THE CONTACT IS WRITTEN **WITH** BARYCENTRIC (:1270-1272).  The tetmesh kernel
//   uses writeRigidVsDeformableContactNoBarycentric; this one uses
//   writeRigidVsDeformableContact, which additionally stores outBarycentric[index]
//   -- and the barycentric is not a by-product, it is how the solver distributes the
//   impulse over the triangle's three vertices.  Dropping it would leave the cloth
//   receiving contacts that push nothing.
//
//   THERE IS NO ERP HACK.  The tetmesh kernel scales penetrating distances to 40%
//   ("VR @@@ add fake ERP. it overshoots otherwise", :1184).  This kernel writes
//   dists[i] RAW (:1269).  Carrying the tetmesh line across would have been an easy
//   and completely silent behaviour change.
//
//   THE FILTER IS A TRIPLE TEST WITH `&&` (:1235-1238).  All THREE of the
//   triangle's vertices must be filtered against this rigid before the pair is
//   skipped.  `||` would suppress a triangle any one of whose vertices is attached
//   -- the difference shows up exactly at the boundary of an attachment region,
//   which is where cloth-vs-rigid scenes actually live.
//
// ===========================================================================
// PxMeshScale ON THE CLOTH BVH: WELL-DEFINED, AND IT HAD TO BE CHECKED
// ===========================================================================
//
// `TrimeshBvh(const PxgFEMCloth&)` (:611-618) sets trimeshVerts, trimeshTriIndices
// and bv32PackedNodes and says NOTHING about the `scale` member -- while this kernel
// calls createConvex (:635) and getTriBarycentric (:675), both of which use
// `scale.toMat33()`.  That reads like an uninitialised-memory bug, and the whole
// port hangs on it, so it was checked rather than assumed:
//
//     PX_CUDA_CALLABLE PX_FORCE_INLINE PxMeshScale(): scale(1.0f), rotation(PxIdentity) {}
//                                                                 -- PxMeshScale.h:71
//
// `PxMeshScale scale;` is a member of class type with a user-provided default
// constructor, so default-initialising the enclosing object RUNS that constructor.
// The value is IDENTITY, deterministically, and the CUDA is well-defined.  This is
// therefore NOT an upstream bug candidate.
//
// The consequence for the port is that both call sites transform the vertices
// through the IDENTITY matrix `scale.toMat33()`, and this file passes identity scale
// and identity rotation to cvxBvhCreateConvex and cvxBvhTriBarycentric rather than
// skipping the transform.  That is NOT the same as using the raw vertices: the
// transform is `column0*v.x + column1*v.y + column2*v.z` with +0 off-diagonals, which
// is exact on every nonzero component but turns a -0 into +0 wherever a +0 product
// joins the sum -- and the barycentric of a triangle with signed-zero coordinates can
// change its own zero sign with it (P9.94d; npCvxBvhGlue_p94dTriBaryZeroSign).
//
// ===========================================================================
// WHAT IS REUSED
// ===========================================================================
//
//   pxgSlangCvxCoreShape (P9.67b)  ShapeRef / PtsRef / supportLocalQ / contactFaceQ
//   pxgSlangCvxCoreCvx   (P9.69a)  gjkDistanceQ / epaDepthQ / faceClipQ
//   pxgSlangCvxCoreMake  (P9.69b)  Gu::makeConvexShape
//   pxgSlangCvxCoreBvh   (P9.70c)  cvxBvhCreateConvex, the triangle arm
//   pxgSlangDefFilter              dfFindRigidPair, the consolidated find()
//   pxgSlangFEMClothDesc           the PxgFEMCloth mirror
//
// The GJK/EPA/clip layer comes in by #include, not `import`: P9.69a measured that
// putting it behind a `module` took compilation of this layer from 1.07 s to over
// 40 minutes.

import pxgSlangNpAbi;           // struct mirrors and quaternion helpers
import pxgSlangCvxCoreShape;    // ShapeRef / PtsRef / supportLocalQ / contactFaceQ
import pxgSlangFEMClothDesc;    // PxgFEMClothGpu
import pxgSlangDefFilter;       // dfFindRigidPair over PxgRigidFilterPair

#include "pxgSlangCvxCoreCvx.slang"    // gjkDistanceQ / epaDepthQ / faceClipQ

// P9.57: FROM THE ENGINE ENUM's numbering, never a remembered literal.  P9.72 wrote
// 9 for eTETRAHEDRONMESH here (that is eHEIGHTFIELD) and check_geom_type_constants.py
// -- which reads PxGeometry.h and compares -- failed the build on it.
static const uint EGT_SPHERE = 0u;
static const uint EGT_PLANE = 1u;
static const uint EGT_CAPSULE = 2u;
static const uint EGT_BOX = 3u;
static const uint EGT_CONVEXCORE = 4u;
static const uint EGT_CONVEXMESH = 5u;
static const uint EGT_TRIANGLEMESH = 8u;   // PxGeometryType::eTRIANGLEMESH

#include "pxgSlangCvxCoreMake.slang"   // Gu::makeConvexShape, all five arms
#include "pxgSlangCvxCoreBvh.slang"    // cvxBvhCreateConvex (the triangle arm)

// The launch is 32 blocks x 512 threads, both LITERALS at
// PxgNarrowphaseCore.cpp:1186-1187.  The bridge forwards gridDim only and drops
// blockDim, so [numthreads] IS the block: 512 here reproduces the CUDA block
// exactly, and WorkgroupCount().x * 512 reproduces `gridDim.x * blockDim.x`.
// tools/check_block_coverage.py is what holds this to the arithmetic.
static const uint CVXCORE_CLOTH_BLOCK = 512u;
static const uint MAX_CONVEX_CONTACTS = 4u;      // Gu::MAX_CONVEX_CONTACTS
static const uint PXG_BITSHIFT_ELEMENT_ID = 20u; // PxgSimulationCoreDesc.h:41

// ---------------------------------------------------------------------------
// Push constant: the CUDA parameter list EXACTLY (:1196-1210).  Fourteen arguments,
// the last of them PxgFEMContactWriter BY VALUE -- nine device pointers plus
// PxU32 maxNumContacts plus a 4-byte tail pad, 80 bytes (PxgFEMCore.h:361-375).
//
//   ten pointers @0..@72, two PxU32 @80/@84, one pointer @88, the writer at @96
//   (8-aligned, so paramAlign moves nothing) -> a 176-byte blob.
// ---------------------------------------------------------------------------
struct CvxCoreClothParams
{
    uint64_t stackPtr;                // @0    const PxU8*
    uint64_t midphasePairsNum;        // @8    const PxU32*
    uint64_t cmInputs;                // @16
    uint64_t shapes;                  // @24
    uint64_t femClothes;              // @32
    uint64_t shapeToRigidRemapTable;  // @40
    uint64_t transformCache;          // @48
    uint64_t contactDistance;         // @56
    uint64_t restDistance;            // @64
    uint64_t filterPairs;             // @72   const PxgRigidFilterPair*
    uint nbFilterPairs;               // @80
    uint stackSizeBytes;              // @84
    uint64_t stackSizeNeededOnDevice; // @88   PxU32*
    // --- PxgFEMContactWriter, by value (PxgFEMCore.h:361) ---
    uint64_t w_outPoint;              // @96   float4*
    uint64_t w_outNormalPen;          // @104  float4*
    uint64_t w_outBarycentric;        // @112  float4*  (LIVE on this path)
    uint64_t w_outContactInfo;        // @120  PxgFemOtherContactInfo*  (16 B stride)
    uint64_t w_totalContactCount;     // @128  PxU32*
    uint64_t w_contactByX;            // @136  PxU64*
    uint64_t w_tempContactByX;        // @144  PxU32*
    uint64_t w_contactIndexSortedByX; // @152  PxU32*
    uint64_t w_contactSortedByX;      // @160  PxU32*  (unused on this path)
    uint w_maxNumContacts;            // @168  PxU32
    uint _pad172;                     // @172 -> 176
}

[[vk::push_constant]]
ConstantBuffer<CvxCoreClothParams> gCloth;

// PxEncodeClothIndex (PxgSimulationCoreDesc.h:101-108).  NOTE the shift constant:
// PXG_BITSHIFT_ELEMENT_ID, not PXG_BITSHIFT_TET_ID.  Both happen to be 20 today
// (PxgSimulationCoreDesc.h:40-41), which is exactly why each is spelled from its own
// line -- if upstream ever splits them, a remembered 20 would encode cloth triangles
// into the softbody's namespace and nothing would fail loudly.
uint encodeCloth(uint clothId, uint elementId)
{
    return (clothId << PXG_BITSHIFT_ELEMENT_ID) | elementId;
}

// Gu::ConvexShape::support (GuConvexSupport.h:539-540) -- WITH the margin.
float3 cvxCoreClothSupport(ShapeRef s, PtsRef pts, float margin, float3 dir)
{
    return poseTransform(s, supportLocalQ(s, pts, quatRotInvQ(s.poseQ, dir))) + dir * margin;
}

// Gu::ConvexShape::computeBounds (GuConvexSupport.h:554-561).
void cvxCoreClothComputeBounds(ShapeRef s, PtsRef pts, float margin, out float3 mn, out float3 mx)
{
    const float3 X = float3(1, 0, 0), Y = float3(0, 1, 0), Z = float3(0, 0, 1);
    mn = float3(cvxCoreClothSupport(s, pts, margin, -X).x,
                cvxCoreClothSupport(s, pts, margin, -Y).y,
                cvxCoreClothSupport(s, pts, margin, -Z).z);
    mx = float3(cvxCoreClothSupport(s, pts, margin, X).x,
                cvxCoreClothSupport(s, pts, margin, Y).y,
                cvxCoreClothSupport(s, pts, margin, Z).z);
}

// PxComputeBarycentric and TrimeshBvh::getTriBarycentric live in pxgSlangCvxCoreBvh
// (cvxBvhComputeBarycentric / cvxBvhTriBarycentric) since P9.94d, so that npCvxBvhGlue
// drives the same text bitwise.

// ===========================================================================
// Gu::generateContacts, the SIX-argument overload (GuConvexSupport.h:1203-1207),
// which forwards PxVec3(0) as the cull direction to the seven-argument one.  The
// cull test `cullDir.dot(axis) < -testEps` is therefore always false here and the
// branch is written out rather than deleted, the same way cvxCoreCvxNpBda and
// cvxCoreTetNpBda do it.
//
// THE TRIMESH KERNEL IS THE ONE THAT DIFFERS: it passes getTriNormal(triIndex) as a
// real cull direction and the test is live there.  This kernel does not, so no
// triangle normal is ever computed on this path.
// ===========================================================================
uint cvxCoreClothGenerateContacts(ShapeRef shA, PtsRef ptsA, float marginA,
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

// PxgFEMContactWriter::writeRigidVsDeformableContact, the BARYCENTRIC variant
// (PxgFEMCore.h:485-499).
//
// It is writeContactCore (:411-427) -- contactByX, tempContactByX,
// contactIndexSortedByX, outPoint, outNormalPen -- followed by outBarycentric and the
// four PxgFemOtherContactInfo fields.  The NoBarycentric sibling used by
// cvxCoreTetNpBda writes the same set MINUS outBarycentric.
//
// setRigidMaterialIndex(mat) followed by markInCollision(false) collapses to a single
// store of (mat & 0x7fffffff), the same way P9.72 established for the tet path.
// contactSortedByX is NOT written here (that is writeRigidVsDeformableContact32's
// job), and totalContactCount is the CALLER's to reserve.
void clothWriteRigidVsDeformable(uint index, float4 contact, float4 normalPen, float4 barycentric,
                                 uint2 pairInd0, uint pairInd1, uint2 rigidId, uint rigidMatId)
{
    if (index >= gCloth.w_maxNumContacts)
        return;
    Ptr<uint2>(gCloth.w_contactByX)[index] = rigidId;
    Ptr<uint>(gCloth.w_tempContactByX)[index] = rigidId.x;   // PxU32(rigidId & 0xffffffff)
    Ptr<uint>(gCloth.w_contactIndexSortedByX)[index] = index;
    Ptr<float4>(gCloth.w_outPoint)[index] = contact;
    Ptr<float4>(gCloth.w_outNormalPen)[index] = normalPen;
    Ptr<float4>(gCloth.w_outBarycentric)[index] = barycentric;
    // PxgFemOtherContactInfo (16 B): pairInd0 u64 @0, pairInd1 u32 @8, rigidMatInd @12.
    const uint64_t ci = gCloth.w_outContactInfo + uint64_t(index) * 16UL;
    Ptr<uint2>(ci)[0] = pairInd0;
    Ptr<uint>(ci + 8UL)[0] = pairInd1;
    Ptr<uint>(ci + 12UL)[0] = rigidMatId & 0x7fffffffu;
}

// ===========================================================================
[shader("compute")]
[numthreads(CVXCORE_CLOTH_BLOCK, 1, 1)]
void convexCoreClothmeshNphase_Kernel32(uint3 dtid : SV_DispatchThreadID)
{
    const uint threadIndex = dtid.x;                                    // :1213
    const uint numThreads = WorkgroupCount().x * CVXCORE_CLOTH_BLOCK;   // :1214

    // stackSize is in uint4 UNITS, not bytes (:1215) -- `stackSizeBytes / sizeof(uint4)`.
    const uint stackSize = gCloth.stackSizeBytes / 16u;
    const uint numMidphasePairsFound = Ptr<uint>(gCloth.midphasePairsNum)[0];
    // The CLAMP is the overflow contract: the midphase may have COUNTED more pairs
    // than it had room to store, and the excess simply is not there.  The count that
    // goes to the atomicMax tail below is the UNCLAMPED one, which is how the engine
    // learns it needs a bigger stack next frame.
    const uint numPairs = min(numMidphasePairsFound, stackSize);
    Ptr<uint4> pairs = Ptr<uint4>(gCloth.stackPtr);

    Ptr<PxgShapeGpu> shapesP = Ptr<PxgShapeGpu>(gCloth.shapes);
    Ptr<PxsCachedTransformGpu> xfP = Ptr<PxsCachedTransformGpu>(gCloth.transformCache);

    for (uint pairIndex = threadIndex; pairIndex < numPairs; pairIndex += numThreads)
    {
        const uint4 testPair = pairs[pairIndex];
        const uint cmIndex = testPair.x;
        const uint triIndex = testPair.y;

        // ---- TestInput (:66-96) + checkTypes (:98-108) ----------------------
        const PxgContactManagerInputGpu cin = Ptr<PxgContactManagerInputGpu>(gCloth.cmInputs)[cmIndex];
        uint cacheRef0 = cin.transformCacheRef0;
        uint cacheRef1 = cin.transformCacheRef1;
        PxgShapeGpu shape0 = shapesP[cin.shapeRef0];
        PxgShapeGpu shape1 = shapesP[cin.shapeRef1];

        // checkTypes(eCONVEXCORE, eTRIANGLEMESH).  The swap is live: the `flip` at
        // PxgNphaseImplementationContext.cpp:1602 is local to the bucket selection and
        // registerContactManagerInternal gets the UNFLIPPED workUnit, so a pair can
        // arrive mesh-first.
        if (shape1.type == EGT_CONVEXCORE && shape0.type == EGT_TRIANGLEMESH)
        {
            const PxgShapeGpu ts = shape0; shape0 = shape1; shape1 = ts;
            const uint tc = cacheRef0; cacheRef0 = cacheRef1; cacheRef1 = tc;
        }

        // ---- TrimeshBvh(femClothes[clothId]) (:611-618) ----------------------
        // Only three of its methods are reachable from here: getTriIndices,
        // createConvex and getTriBarycentric.  getNodes and scaleBounds are never
        // called -- there is no traversal on this path.
        const uint clothId = shape1.particleOrSoftbodyId;
        const PxgFEMClothGpu cloth = Ptr<PxgFEMClothGpu>(gCloth.femClothes)[clothId];
        Ptr<float4> clothVerts = Ptr<float4>(cloth.mPosition_InvMass);
        Ptr<uint4> clothTriInds = Ptr<uint4>(cloth.mTriangleVertexIndices);
        // The scale left IDENTITY by that constructor -- see the header.  Spelled out,
        // and pushed through the identity matrix at BOTH call sites below exactly as the
        // CUDA does (the transform is not a no-op on signed zeros).
        const float3 clothScale = float3(1, 1, 1);
        const float4 clothScaleRot = float4(0, 0, 0, 1);

        // ---- the triple filter (:1233-1239) ----------------------------------
        // rigidId.getInd() is the whole 64-bit word; dfFindRigidPair takes it as the
        // uint2 the corpus carries it in.
        const uint64_t rigidIdW = Ptr<uint64_t>(gCloth.shapeToRigidRemapTable)[cacheRef0];
        const uint2 rigidId = uint2(uint(rigidIdW & 0xFFFFFFFFUL), uint(rigidIdW >> 32));

        const uint4 vertInds = clothTriInds[triIndex];
        // `&&` AND NOT `||`: the pair is skipped only when ALL THREE vertices are
        // filtered against this rigid.  Slang's && short-circuits exactly as C++'s
        // does, so the second and third searches are skipped when the first misses --
        // same work, same result.
        if (dfFindRigidPair(gCloth.filterPairs, gCloth.nbFilterPairs, rigidId,
                            encodeCloth(clothId, vertInds.x)) &&
            dfFindRigidPair(gCloth.filterPairs, gCloth.nbFilterPairs, rigidId,
                            encodeCloth(clothId, vertInds.y)) &&
            dfFindRigidPair(gCloth.filterPairs, gCloth.nbFilterPairs, rigidId,
                            encodeCloth(clothId, vertInds.z)))
            continue;

        // ---- the two shapes (:1241-1256) -------------------------------------
        const PxsCachedTransformGpu xf0 = xfP[cacheRef0];
        const PxsCachedTransformGpu xf1 = xfP[cacheRef1];
        const float4 q0 = float4(xf0.qx, xf0.qy, xf0.qz, xf0.qw);
        const float3 p0 = float3(xf0.px, xf0.py, xf0.pz);
        const float4 q1 = float4(xf1.qx, xf1.qy, xf1.qz, xf1.qw);
        const float3 p1 = float3(xf1.px, xf1.py, xf1.pz);

        const float contactDist = Ptr<float>(gCloth.contactDistance)[cacheRef0]
                                + Ptr<float>(gCloth.contactDistance)[cacheRef1];
        // restDist = restDistance ? restDistance[testIndex] : 0 (:94).  The index is
        // the CONTACT MANAGER index, not the pair index -- restDistance is per contact
        // manager, and the TestInput ctor is handed cmIndex (:1229).
        const float restDist = Ptr<float>(gCloth.restDistance)[cmIndex];

        // transform0in1 = transform1.transformInv(transform0) (:1241).
        const float4 conjQ1 = float4(-q1.x, -q1.y, -q1.z, q1.w);
        const float4 aToBq = npAbiQuatMul(conjQ1, q0);
        // P9.94d: qinv.rotate(src.p - p) with qinv = q.getConjugate(), as
        // PxTransform::transformInv(PxTransform) spells it (PxTransform.h:162-169) --
        // not q1.rotateInv(.), which differs only in the sign of an exact zero.
        // NO FIXTURE PINS THIS LINE, AND THAT IS MEASURED, NOT ASSUMED.  aToBp only
        // becomes ShapeRef.poseP, which every consumer uses as an ADDEND (poseTransform
        // `quatRotQ(q, v) + poseP`, the face points `(quatRotQ(q, fpl) + poseP) + wn *
        // margin`, the support `... + dir * margin`) or a subtrahend (`point - poseP`).
        // A zero's sign survives a sum only when the other addend is -0 as well, so it
        // reaches an output only along an all-zero chain.  cvxCoreClothNp's box at the
        // origin gives the same bits on both spellings; a probe (P9.94d, not committed)
        // with a POINT core (margin 0.2) at (-0, 1.05, -0) over the same cloth -- where
        // the old spelling gives aToBp.z = +0 and this one -0 -- emitted bit-identical
        // points, normals and barycentrics both ways.  Revert mutants survive
        // cvxCoreClothNp for that reason (and because the cloth's own transform is
        // identity, so a dropped/doubled conjugate is a zero-sign change too); the
        // literal spelling is kept because it is what the CUDA computes.
        const float3 aToBp = npAbiQuatRotate(conjQ1, p0 - p1);

        ShapeRef convex;
        float convexMargin;
        PtsRef convexPts;
        const bool okConvex = makeConvexShapeCvxCore(shape0, aToBq, aToBp,
                                                     convex, convexMargin, convexPts);
        // The CUDA has no such gate -- Gu::makeConvexShape returns void and the kernel
        // asserts `convex0.isValid()` (:1256), which is a no-op in Release.  The
        // corpus's port returns false for a core type it cannot build, and every other
        // convex-core kernel gates on it the same way (cvxCoreCvxNpBda:198
        // `if (okA && okB)`, cvxCoreTetNpBda's `hasNode = okConvex`).  Skipping is what
        // the assert intends; proceeding would run GJK on a zeroed ShapeRef.
        if (!okConvex)
            continue;

        // ---- bounds0 = convex0.computeBounds(); fattenFast(contactDist) (:1243-1244)
        // Unlike every other kernel in the family, this box is NOT a query box -- the
        // midphase already chose the triangle.  Its ONLY use is the margin below.
        float3 cbMin, cbMax;
        cvxCoreClothComputeBounds(convex, convexPts, convexMargin, cbMin, cbMax);
        cbMin -= float3(contactDist, contactDist, contactDist);
        cbMax += float3(contactDist, contactDist, contactDist);

        // "let's give triangle some margin proportional to the convex shape's size -
        // for stability" (:1246-1250).  maxElement of the FATTENED dimensions.
        const float triMarginK = 0.001f;
        const float3 dims = cbMax - cbMin;
        const float margin = max(dims.x, max(dims.y, dims.z)) * triMarginK;

        const float3 tv0 = clothVerts[vertInds.x].xyz;
        const float3 tv1 = clothVerts[vertInds.y].xyz;
        const float3 tv2 = clothVerts[vertInds.z].xyz;

        ShapeRef tri;
        PtsRef triPts;
        float triMargin;
        cvxBvhCreateConvex(clothScale, clothScaleRot, tv0, tv1, tv2, margin,
                           tri, triPts, triMargin);

        // ---- generateContacts (:1258) ----------------------------------------
        float3 normal;
        float4 patch[MAX_CONVEX_CONTACTS];
        const uint numPoints = cvxCoreClothGenerateContacts(convex, convexPts, convexMargin,
                                                            tri, triPts, triMargin,
                                                            contactDist, normal, patch);

        // ---- the emit (:1260-1274) -------------------------------------------
        // NO warp aggregation: the CUDA's atomicAdd is inside `if (numPoints)`, so
        // every contributing thread reserves its OWN run of slots independently.
        if (numPoints != 0u)
        {
            const uint pairInd1 = encodeCloth(clothId, triIndex);
            uint index;
            InterlockedAdd(Ptr<uint>(gCloth.w_totalContactCount)[0], numPoints, index);
            for (uint i = 0u; i < MAX_CONVEX_CONTACTS; ++i)
            {
                if (i >= numPoints) break;
                // point = points[i] + normal * dists[i] * 0.5f (:1267): the contact is
                // placed HALFWAY along the penetration, not on either surface.
                const float3 point = patch[i].xyz + normal * patch[i].w * 0.5f;
                const float4 posRest = float4(point, restDist);
                // THE NORMAL IS NEGATED ON WRITE (:1269), and the distance is written
                // RAW.  The tetmesh kernel's `if (d < 0) d *= 0.4f` ERP hack is NOT
                // here and must not be carried across.
                const float4 normDist = float4(-normal.x, -normal.y, -normal.z, patch[i].w);
                // The barycentric is of the MIDPOINT `point`, not of patch[i].xyz
                // (:1270) -- it is computed after the half-penetration shift.
                // bvh.getTriBarycentric (:672-680) transforms the three vertices by
                // scale.toMat33() FIRST, like createConvex; P9.94d made the port do the
                // same instead of handing it the raw vertices (they differ on -0
                // coordinates -- see the header).  The function is pinned bitwise by
                // npCvxBvhGlue (incl. the zero-sign witnesses); reverting THIS call to
                // raw vertices survives cvxCoreClothNp, whose cloth triangles are
                // horizontal with no -0 coordinate -- an exhaustive host search over
                // that family (x,z in {+-0.5,+-0}, contact point at a vertex or grid
                // point) found no barycentric word that the identity transform changes.
                const float4 bary = cvxBvhTriBarycentric(clothScale, clothScaleRot,
                                                         tv0, tv1, tv2, point);
                clothWriteRigidVsDeformable(index + i, posRest, normDist, bary,
                                            rigidId, pairInd1, rigidId, shape0.materialIndex);
            }
        }
    }

    // :1277-1278.  The UNCLAMPED count, in BYTES, so the engine can grow the stack.
    // Guarded on the GLOBAL thread index, not the group-local one: with 32 groups a
    // `gtid.x == 0` guard would run it 32 times.  Harmless for a max, but not what
    // the CUDA says, and the next person to add a non-idempotent line here would
    // inherit the bug.
    if (threadIndex == 0u)
    {
        uint old;
        InterlockedMax(Ptr<uint>(gCloth.stackSizeNeededOnDevice)[0],
                       numMidphasePairsFound * 16u, old);
    }
}
