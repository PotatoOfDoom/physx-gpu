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

// cvxCoreCvxNpBda.slang -- `convexCoreConvexNphase_Kernel`
// (convexCoreCollision.cu:497-545) on the engine BDA ABI.  P9.69b.
//
// The second of the five convexCore*Nphase kernels, and the one that lets a
// PxConvexCoreGeometry rest on ANY ordinary convex: a sphere, a capsule, a box, a
// cooked convex mesh, or another convex core (PxgNphaseImplementationContext.cpp
// gTable, GPU_BUCKET_ID::eConvexCoreConvex).
//
// ===========================================================================
// ALMOST NOTHING HERE IS NEW, AND THAT IS THE POINT
// ===========================================================================
//
//   pxgSlangCvxCoreShape (P9.67b)  ShapeRef / supportLocalQ over all seven core
//                                  types / contactFaceQ / reducePolygonQ.
//   pxgSlangCvxCoreCvx   (P9.69a)  gjkDistanceQ / epaDepthQ / faceClipQ -- the
//                                  GJK simplex walk, the EPA polytope and the
//                                  face clipper, lifted out of npCvxCoreCvxGen
//                                  so that the fixture which verifies them
//                                  BITWISE against a CPU mirror verifies THESE
//                                  instructions and not a copy of them.
//   pxgSlangCvxCoreMake  (P9.69b)  Gu::makeConvexShape, all five arms.
//   pxgSlangCvxCoreOut   (P9.69b)  TestOutput::writeOutput, shared with the
//                                  plane kernel rather than transcribed twice
//                                  (P9.37: replicated tails drift in
//                                  ASSOCIATION, invisibly).
//
// What is genuinely new is the parameter plumbing, the recentring shift, and the
// decision of which of those four each line belongs to.
//
// ===========================================================================
// THE SHIFT, WHICH IS NOT DECORATION
// ===========================================================================
//
//     const PxVec3 shift = (in.transform0.p + in.transform1.p) * 0.5f;
//     const PxTransform pose0(in.transform0.p - shift, in.transform0.q);
//     const PxTransform pose1(in.transform1.p - shift, in.transform1.q);
//     ...
//     contacts[i] = PxContact({ points[i] + shift, dists[i] });
//
// (:525-538).  GJK and EPA both accumulate differences of support points; running
// them about the pair's midpoint keeps those differences from being computed as
// the difference of two large world coordinates.  A port that drops the shift
// gets the same answer near the origin and loses precision far from it -- which
// is exactly the class of defect a scene near the origin cannot see, so it is
// reproduced here literally rather than "simplified away".
//
// ===========================================================================
// THE ePOINTS ARM IS REACHABLE HERE, UNLIKE IN THE PLANE KERNEL
// ===========================================================================
//
// cvxCorePlaneNpBda documents its slot 1 as never being eCONVEXMESH.  This
// kernel's either slot can be, and until P9.69b the shared shape layer took hull
// points as a `float4 pool[12]` BY VALUE -- a cooked hull carries up to 255.  The
// layer reads them through Ptr<float4> now and makeConvexShapeCvxCore hands back
// the hull's device address, so the arm is written rather than excused.
//
// ===========================================================================
// THE ABI ROW IS SHARED WITH convexCorePlaneNphase_Kernel
// ===========================================================================
//
// Both kernels take the SAME eighteen arguments in the same order
// (PxgNarrowphaseCore.cpp:826-846 and :896-916 differ only in the block size), so
// the 136-byte kCvxCorePlane row is reused rather than copied.  P9.48d is why the
// row has to be exact rather than close: a kernel whose ABI row is wrong gets
// `pushConstantSizeFor() == 0` and every launch REFUSED IN SILENCE, with
// fixtures, manifest test, suite and scenes all still green.
//
// ===========================================================================
// THE GROUP IS 64 AND THE ENGINE'S BLOCK IS 512, SO THERE IS A COVERING LOOP
// ===========================================================================
//
// numBlocks = ceil(numTests / 512) (PxgNarrowphaseCore.cpp:917-921) and the
// bridge forwards gridDim only, so a flat 64-wide pass would reach one eighth of
// the pairs on every call.  That is the P9.68 defect, and tools/check_block_coverage.py
// now fails the build on it.  The stride is the port's own launched thread count.

import pxgSlangNpAbi;           // the struct mirrors and quaternion helpers
import pxgSlangCvxCoreShape;    // ShapeRef / supportLocalQ / contactFaceQ
#include "pxgSlangCvxCoreCvx.slang"    // gjkDistanceQ / epaDepthQ / faceClipQ
#include "pxgSlangCvxCoreOut.slang"    // the shared writeOutput tail + its constants

// The group size, named once so the stride below cannot drift from the
// [numthreads] that produced it.
static const uint CVXCORE_CVX_BLOCK = 64u;

// P9.57: FROM THE ENGINE ENUM's numbering, never a remembered literal.
// tools/check_geom_type_constants.py reads PxGeometry.h and checks these.
static const uint EGT_SPHERE = 0u;
static const uint EGT_PLANE = 1u;
static const uint EGT_CAPSULE = 2u;
static const uint EGT_BOX = 3u;
static const uint EGT_CONVEXCORE = 4u;
static const uint EGT_CONVEXMESH = 5u;

#include "pxgSlangCvxCoreMake.slang"   // Gu::makeConvexShape, all five arms

// ---------------------------------------------------------------------------
// Push constant: the CUDA __global__ parameter list EXACTLY
// (convexCoreCollision.cu:497-516).  Byte-identical to the plane kernel's, which
// is why they share an ABI row.
// ---------------------------------------------------------------------------
struct CvxCoreCvxParams
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
ConstantBuffer<CvxCoreCvxParams> gParams;

// ---------------------------------------------------------------------------
// the per-test body
// ---------------------------------------------------------------------------
void convexCoreConvexOneTest(uint testIndex)
{
    // ---- TestInput (:66-96) -------------------------------------------------
    Ptr<PxgContactManagerInputGpu> cmInputsP = Ptr<PxgContactManagerInputGpu>(gParams.cmInputs);
    const PxgContactManagerInputGpu cin = cmInputsP[testIndex];

    Ptr<PxgShapeGpu> shapesP = Ptr<PxgShapeGpu>(gParams.shapes);
    const PxgShapeGpu shape0 = shapesP[cin.shapeRef0];
    const PxgShapeGpu shape1 = shapesP[cin.shapeRef1];
    const uint cacheRef0 = cin.transformCacheRef0;
    const uint cacheRef1 = cin.transformCacheRef1;

    // NO checkTypes SWAP HERE, and that is faithful: convexCoreConvexNphase_Kernel
    // constructs its TestInput and goes straight to makeConvexShape on both slots
    // (:523-531).  The plane kernel needs a swap because ONE of its slots must be
    // the plane; here both slots are convexes and either order is meaningful.
    Ptr<PxsCachedTransformGpu> xfP = Ptr<PxsCachedTransformGpu>(gParams.transformCache);
    const PxsCachedTransformGpu xf0 = xfP[cacheRef0];
    const PxsCachedTransformGpu xf1 = xfP[cacheRef1];

    const float contactDist = Ptr<float>(gParams.contactDistance)[cacheRef0]
                            + Ptr<float>(gParams.contactDistance)[cacheRef1];

    const float3 p0 = float3(xf0.px, xf0.py, xf0.pz);
    const float3 p1 = float3(xf1.px, xf1.py, xf1.pz);
    const float3 shift = (p0 + p1) * 0.5f;      // (:525)

    const float4 q0 = float4(xf0.qx, xf0.qy, xf0.qz, xf0.qw);
    const float4 q1 = float4(xf1.qx, xf1.qy, xf1.qz, xf1.qw);

    ShapeRef shA, shB;
    float margin0, margin1;
    PtsRef ptsA, ptsB;
    const bool okA = makeConvexShapeCvxCore(shape0, q0, p0 - shift, shA, margin0, ptsA);
    const bool okB = makeConvexShapeCvxCore(shape1, q1, p1 - shift, shB, margin1, ptsB);

    uint nbContacts = 0u;
    float3 pointsOut[4] = { float3(0,0,0), float3(0,0,0), float3(0,0,0), float3(0,0,0) };
    float distsOut[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float3 normal = float3(0, 0, 0);

    if (okA && okB)
    {
        // ---- Gu::generateContacts(ConvexShape, ConvexShape)
        // (GuConvexSupport.h:1169-1201), transcribed from npCvxCoreCvxGen's entry,
        // which is the fixture that verifies it.
        const float maxDist = margin0 + margin1 + contactDist;

        float dist; uint tagG;
        float3 point0, point1, axis;
        gjkDistanceQ(shA, shB, ptsA, ptsB, maxDist, dist, tagG, point0, point1, axis);

        if (dist < FLT_EPS)
        {
            uint tagE;
            epaDepthQ(shA, shB, ptsA, ptsB, dist, tagE, point0, point1, axis);
        }

        // THE CULL DIRECTION IS ZERO ON THIS PATH.  The fixture carries a cullDir
        // because GuConvexSupport's generateContacts takes one; the kernel calls
        // the SIX-argument overload (:537), which passes PxVec3(0) -- and
        // `dot(0, axis) < -1e-5` is false, so nothing is culled.  Writing the test
        // out and letting it be false is closer to the .cu than deleting it.
        bool emit = !(dist > maxDist);

        if (emit)
        {
            point0 -= axis * margin0;
            point1 += axis * margin1;
            dist -= margin0 + margin1;

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
                wfp0[i] = i < np0 ? (quatRotQ(shA.poseQ, fpl[i]) + shA.poseP) + wn0 * margin0 : float3(0, 0, 0);

            const uint np1 = contactFaceQ(shB, ptsB, quatRotInvQ(shB.poseQ, axis),
                                          quatRotInvQ(shB.poseQ, point1 - shB.poseP), fn1, fpl, trig1);
            const float3 wn1 = quatRotQ(shB.poseQ, fn1);
            for (uint i = 0u; i < 4u; ++i)
                wfp1[i] = i < np1 ? (quatRotQ(shB.poseQ, fpl[i]) + shB.poseP) + wn1 * margin1 : float3(0, 0, 0);

            float4 patch[4];
            nbContacts = faceClipQ(wfp0, np0, wn0, wfp1, np1, wn1, point0, point1, axis, normal, patch);

            // contacts[i] = PxContact({ points[i] + shift, dists[i] }) (:537-538):
            // the SHIFT COMES BACK HERE, and only here.
            for (uint i = 0u; i < 4u; ++i)
            {
                pointsOut[i] = i < nbContacts ? patch[i].xyz + shift : float3(0, 0, 0);
                distsOut[i] = i < nbContacts ? patch[i].w : 0.0f;
            }
        }
    }

    // The kernel passes `normal` -- NOT its negation -- to writeOutput (:540).
    // The plane kernel passes -normal because its generator's normal points out of
    // the plane; here faceClipQ already returns the axis in the convention
    // writeOutput wants.  Getting this backwards flips every contact patch and is
    // silent: the bodies still collide, they are just pushed the wrong way.
    const float3 patchNormal = normal;

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

    cvxCoreWriteOutput(testIndex, shape0, shape1, patchNormal, pointsOut, distsOut,
                       nbContacts, orefs);
}

// `threadIndex / 1; // 1 tread per test` (:517-518), upstream's typo kept.
[shader("compute")]
[numthreads(CVXCORE_CVX_BLOCK, 1, 1)]
void convexCoreConvexNphase_Kernel(uint3 dtid : SV_DispatchThreadID)
{
    const uint stride = WorkgroupCount().x * CVXCORE_CVX_BLOCK;
    for (uint testIndex = dtid.x; testIndex < gParams.numTests; testIndex += stride)
        convexCoreConvexOneTest(testIndex);
}
