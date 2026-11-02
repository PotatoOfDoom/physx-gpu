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

// cvxCorePlaneNpBda.slang -- `convexCorePlaneNphase_Kernel`
// (convexCoreCollision.cu:449-494) on the engine BDA ABI.  P9.67c.
//
// The first of the five convexCore*Nphase kernels, and the one that makes
// PXG_SCENE=convexcoreplane -- added and measured in P9.67a -- stop dropping its
// cylinder through the floor:
//
//     [pxg-unmapped] convexCorePlaneNphase_Kernel was launched 183 time(s) but
//     has no SPIR-V mapping: every launch was SKIPPED
//     [convexCoreCylinderOnPlane] final y=-73.8070 (rest 0.500), |v|=39.2401
//
// ===========================================================================
// WHERE THE CODE CAME FROM, AND WHAT IS NEW HERE
// ===========================================================================
//
// Almost none of the arithmetic is new.  Three verified bodies are assembled:
//
//   pxgSlangCvxCoreShape (P9.67b)  ShapeRef, supportLocalQ over all seven core
//                                  types, contactFaceQ, reducePolygonQ -- the
//                                  617 lines moved out of npCvxCorePlaneGen so
//                                  that its fixture goes on testing THIS code
//                                  rather than a copy of it;
//   npCvxCorePlaneGen              the plane driver itself
//                                  (Gu::generateContacts(PxPlane, ConvexShape),
//                                  GuConvexSupport.cpp:41-75), transcribed from
//                                  the fixture entry it verifies;
//   npCvxPlaneNphase               the contact/patch output tail, which is the
//                                  same TestOutput::writeOutput skeleton
//                                  (convexCoreCollision.cu:177-290) already on
//                                  the engine ABI for the convex-vs-plane pair.
//
// What is genuinely new is the parameter plumbing, makeConvexShape, and the
// decision of which of those three each line belongs to.
//
// ===========================================================================
// makeConvexShape: THE CORE TYPE IS SMUGGLED THROUGH hullOrMeshPtr
// ===========================================================================
//
// convexCoreCollision.cu:115-170 turns a PxgShape into a Gu::ConvexShape, and the
// eCONVEXCORE case is not a pointer dereference at all:
//
//     convex.coreType = Gu::ConvexCore::Type::Enum(shape.hullOrMeshPtr);
//     memcpy(convex.coreData, &shape.scale.scale.x, MAX_CORE_SIZE);
//     convex.margin = shape.scale.rotation.w;
//
// The 64-bit hull POINTER field carries a small ENUM, the core's parameters live
// in the PxMeshScale's scale bytes, and the margin in the scale rotation's w.  A
// port that reads hullOrMeshPtr as an address here dereferences a number between
// 0 and 5.
//
// The other four cases matter too, because this kernel's slot 1 is whatever the
// pair table sent: a SPHERE becomes ePOINT with margin = radius, a CAPSULE
// becomes eSEGMENT with length = 2*halfHeight and margin = radius, a BOX becomes
// eBOX with extents = 2*scale and margin 0.  Only eCONVEXMESH (ePOINTS, reading
// the cooked hull) is absent here -- convex-vs-plane has its own kernel.
//
// ===========================================================================
// ONE THREAD PER TEST
// ===========================================================================
//
// `const PxU32 testIndex = threadIndex / 1; // 1 tread per test` (:471, the
// typo is upstream's).  So unlike convexPlaneNphase_Kernel -- a warp per test,
// [numthreads(32,4,1)] -- this is a flat one-thread-per-test kernel and the
// output tail runs entirely on the owning thread.  That tail was already written
// serially in npCvxPlaneNphase (a mask and a loop, not warp primitives), which is
// why it transplants rather than needing rework.
//
// THE CUDA HAS NO GRID-STRIDE LOOP.  THE PORT NEEDS ONE ANYWAY (P9.68).
//
// The CUDA body really is a single `if (testIndex >= numTests) return;` guard
// (:473-474) with no loop, and reading only the .cu that reads as "the host
// launch covers numTests exactly, so a flat port is faithful".  It is not,
// and the reason is one the .cu cannot show:
//
//     host   (PxgNarrowphaseCore.cpp:848-852)
//              numThreadsPerBlock = 1024
//              numBlocks          = ceil(numTests / 1024)
//     bridge (PxgVulkanCudaContext::launchKernel)
//              forwards gridDim ONLY -- blockDim's parameter names are
//              commented out -- so numBlocks becomes the workgroup count and
//              THE SLANG numthreads IS THE BLOCK SIZE.
//     result   ceil(numTests/1024) * 64 threads, i.e. one sixteenth of the
//              tests, on EVERY call.  Not a ceiling past which work is lost:
//              15/16 of the pairs are dropped at any scene size above 64.
//
// This is the P9.61 defect verbatim -- zeroNormals/normalizeNormals declared
// 256 against a host block of 1024 and reached a quarter of the guide vertices
// forever.  The first version of this shader had it and shipped, because the
// convexcoreplane scene has ONE pair and no fixture can see a divisor.
// tools/check_block_coverage.py is the FOURTEENTH checker and exists so that
// the next one fails at the build instead of at a scene nobody wrote.
//
// The stride is the port's own, never the CUDA's: `WorkgroupCount().x * BLOCK`
// is the number of threads ACTUALLY launched, so it covers whatever the host
// meant by its own divisor.  `gridDim.x * blockDim.x` has no spelling here.
//
// ===========================================================================
// THE ABI ROW IS ITS OWN -- 136 BYTES, NOT THE CONVEX-PLANE 144
// ===========================================================================
//
// convexCorePlaneNphase_Kernel takes 18 arguments; convexPlaneNphase_Kernel takes
// 20.  The two extra are `cmPersistentManifolds` and `toleranceLength`, and this
// kernel has neither -- it keeps no manifold and needs no margin scale.  So the
// blob is 136 bytes and the row is NOT shared, however similar the two look.
// P9.48d is the reason to be exact rather than close: a kernel whose ABI row is
// wrong gets `pushConstantSizeFor() == 0` and every launch REFUSED IN SILENCE,
// with fixtures, manifest test, suite and scenes all still green.

import pxgSlangNpAbi;           // the struct mirrors and quaternion helpers
import pxgSlangCvxCoreShape;    // P9.67b: ShapeRef / supportLocalQ / contactFaceQ
#include "pxgSlangCvxCoreOut.slang"   // P9.69b: the shared writeOutput tail

// The group size, named once so the stride below cannot drift from the
// [numthreads] that produced it.  Its VALUE is this port's choice; what makes
// the kernel correct is the stride derived from it, not agreement with the
// engine's 1024 (see the header).  The launch site (PxgNarrowphaseCore.cpp,
// PXG_NO_CUDA_TOOLKIT) launches 1024 / CVXCORE_PLANE_BLOCK groups per CUDA block,
// so the flat thread index is the CUDA's and each thread takes the test of its
// CUDA thread; the threads never cooperate (one test each, the output tail on the
// owning thread, no shared memory).  Before, the group ran two of the block's 32
// warps and the other thirty as further passes of the loop.  Widening the group
// to the block instead was tried and dropped: every thread carries the
// support/clip state in private arrays, and on llvmpipe the first launch of the
// 512-wide convex-vs-convex sibling took 29 s against 0.1 s at 64.
static const uint CVXCORE_PLANE_BLOCK = 64u;
// P9.57: FROM THE ENGINE ENUM's numbering, never a remembered literal.
// tools/check_geom_type_constants.py reads PxGeometry.h and checks these.
static const uint EGT_SPHERE = 0u;
static const uint EGT_PLANE = 1u;
static const uint EGT_CAPSULE = 2u;
static const uint EGT_BOX = 3u;
static const uint EGT_CONVEXCORE = 4u;
static const uint EGT_CONVEXMESH = 5u;

// ---------------------------------------------------------------------------
// Push constant: the CUDA __global__ parameter list EXACTLY
// (convexCoreCollision.cu:449-468).  Natural alignment: numTests(4) @0, the
// pointer bank starts @8, then the three PxU32 byte limits, rounded to maxAlign 8.
// ---------------------------------------------------------------------------
struct CvxCorePlaneParams
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
ConstantBuffer<CvxCorePlaneParams> gParams;

// PxQuat::getBasisVector0 (PxQuat.h:260-265) is its OWN formula, not
// rotate(q,(1,0,0)):  x2 = 2x, w2 = 2w, then
//   ( w*w2 - 1 + x*x2,  z*w2 + y*x2,  -y*w2 + z*x2 )
// The x component is `2w^2 - 1 + 2x^2` against a generic rotate's
// `(w^2 - u.u) + 2x^2`.  Equal only because a unit quaternion has
// w^2+x^2+y^2+z^2 = 1, and not the same float expression.
float3 quatBasisVector0Np(float4 q)
{
    const float x2 = q.x * 2.0f;
    const float w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + (q.x * x2),
                  (q.z * w2) + (q.y * x2),
                  (-q.y * w2) + (q.z * x2));
}


// Gu::makeConvexShape (convexCoreCollision.cu:115-170).  P9.69b moved it into a
// shared include so the convex arm does not get a second transcription; it now
// carries the ePOINTS case too, which THIS kernel's slot 1 never presents (convex
// mesh vs plane has its own kernel) but which cvxCoreCvxNpBda's does.
#include "pxgSlangCvxCoreMake.slang"

// ---------------------------------------------------------------------------
// the kernel
// ---------------------------------------------------------------------------
// The per-test body, lifted out of the entry point so the covering loop above it
// stays a two-liner and the single CUDA `return` guard needs no rewriting into a
// `continue`.  check_grid_stride.py follows one level of same-file helper calls,
// which is what makes this shape readable to it as well as to a person.
void convexCorePlaneOneTest(uint testIndex)
{
    // ---- TestInput (:66-96) + checkTypes (:98-108) --------------------------
    Ptr<PxgContactManagerInputGpu> cmInputsP = Ptr<PxgContactManagerInputGpu>(gParams.cmInputs);
    const PxgContactManagerInputGpu cin = cmInputsP[testIndex];

    Ptr<PxgShapeGpu> shapesP = Ptr<PxgShapeGpu>(gParams.shapes);
    PxgShapeGpu shape0 = shapesP[cin.shapeRef0];
    PxgShapeGpu shape1 = shapesP[cin.shapeRef1];
    uint cacheRef0 = cin.transformCacheRef0;
    uint cacheRef1 = cin.transformCacheRef1;

    Ptr<PxsCachedTransformGpu> xfP = Ptr<PxsCachedTransformGpu>(gParams.transformCache);
    PxsCachedTransformGpu xf0 = xfP[cacheRef0];
    PxsCachedTransformGpu xf1 = xfP[cacheRef1];

    // checkTypes(ePLANE, eCONVEXCORE): swap when the operands arrived reversed.
    if (EGT_PLANE == shape1.type && EGT_CONVEXCORE == shape0.type)
    {
        const PxgShapeGpu ts = shape0; shape0 = shape1; shape1 = ts;
        const uint tc = cacheRef0; cacheRef0 = cacheRef1; cacheRef1 = tc;
        const PxsCachedTransformGpu tx = xf0; xf0 = xf1; xf1 = tx;
    }

    const float contactDist = Ptr<float>(gParams.contactDistance)[cacheRef0]
                            + Ptr<float>(gParams.contactDistance)[cacheRef1];

    const float4 planeQ = float4(xf0.qx, xf0.qy, xf0.qz, xf0.qw);
    const float3 planeP = float3(xf0.px, xf0.py, xf0.pz);
    const float4 convexQ = float4(xf1.qx, xf1.qy, xf1.qz, xf1.qw);
    const float3 convexP = float3(xf1.px, xf1.py, xf1.pz);

    // PxPlane(in.transform0.p, in.transform0.q.getBasisVector0()) (:478).
    // getBasisVector0 is its OWN formula, not rotate(q,(1,0,0)) -- see
    // pxgSlangCapsuleHfCore's header for why that distinction is load-bearing.
    const float3 pn = quatBasisVector0Np(planeQ);
    const float pd = -dot3Q(planeP, pn);   // PxPlane(point, normal): d = -p.n

    ShapeRef sh;
    float margin;
    PtsRef shPts;
    const bool ok = makeConvexShapeCvxCore(shape1, convexQ, convexP, sh, margin, shPts);

    // ---- Gu::generateContacts(PxPlane, ConvexShape) (GuConvexSupport.cpp:41-75)
    // Transcribed from npCvxCorePlaneGen's entry, which is the fixture that
    // verifies it.  `s.ptsAddr` there is the ePOINTS hull's device address; this
    // kernel's slot 1 is never eCONVEXMESH -- convex-vs-plane has its own kernel --
    // so a null pool is passed and supportLocalQ's CORE_POINTS branch is
    // unreachable here.  It is never DEREFERENCED either: that branch returns
    // before touching the pool when the core is not ePOINTS.
    const PtsRef ptsPool = shPts;   // null unless slot 1 is eCONVEXMESH

    const float3 normal = -pn;

    uint nbContacts = 0u;
    float3 pointsOut[4] = { float3(0,0,0), float3(0,0,0), float3(0,0,0), float3(0,0,0) };
    float distsOut[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    if (ok)
    {
        // ConvexShape::support = pose.transform(supportLocal(pose.rotateInv(dir)))
        //                     + dir * margin
        const float3 point1 = poseTransform(sh, supportLocalQ(sh, ptsPool, quatRotInvQ(sh.poseQ, normal)))
                            + normal * margin;
        const float dist = dot3Q(point1, pn) + pd;   // PxPlane::distance

        if (dist < contactDist)
        {
            // wrapper prologue (GuConvexSupport.h:545)
            const float3 dirL = quatRotInvQ(sh.poseQ, normal);
            const float3 refL = quatRotInvQ(sh.poseQ, point1 - sh.poseP);

            float3 faceNormal;
            float3 fp[4];
            uint trig = 0u;
            const uint numPoints = contactFaceQ(sh, ptsPool, dirL, refL, faceNormal, fp, trig);

            // wrapper epilogue: world rotate + margin push
            const float3 worldNormal = quatRotQ(sh.poseQ, faceNormal);

            if (numPoints == 0u)
            {
                pointsOut[nbContacts] = point1 + normal * dist * 0.5f;
                distsOut[nbContacts] = dist;
                ++nbContacts;
            }

            for (uint i = 0u; i < numPoints; ++i)
            {
                const float3 p1 = (quatRotQ(sh.poseQ, fp[i]) + sh.poseP) + worldNormal * margin;
                const float d = dot3Q(p1, pn) + pd;
                pointsOut[nbContacts] = p1 + normal * d * 0.5f;
                distsOut[nbContacts] = d;
                ++nbContacts;
            }
        }
    }

    // The kernel passes -normal to writeOutput (:492).
    const float3 patchNormal = -normal;

    // ---- TestOutput::writeOutput (:177-290) --------------------------------
    // P9.69b: the tail lives in pxgSlangCvxCoreOut so the convex arm shares these
    // bytes rather than getting a second transcription of them (P9.37).
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

// The engine's block is 1024 and the port launches 1024 / CVXCORE_PLANE_BLOCK
// groups per block, so at that grid the loop makes exactly one pass; it stays for
// any other grid.  `threadIndex / 1; // 1 tread per test` (:470-471) --
// upstream's typo kept -- is why the index is the flat thread id and not a warp.
[shader("compute")]
[numthreads(CVXCORE_PLANE_BLOCK, 1, 1)]
void convexCorePlaneNphase_Kernel(uint3 dtid : SV_DispatchThreadID)
{
    const uint stride = WorkgroupCount().x * CVXCORE_PLANE_BLOCK;
    for (uint testIndex = dtid.x; testIndex < gParams.numTests; testIndex += stride)
        convexCorePlaneOneTest(testIndex);
}
