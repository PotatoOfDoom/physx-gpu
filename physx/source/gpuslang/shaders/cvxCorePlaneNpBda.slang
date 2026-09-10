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

static const uint INVALID = 0xFFFFFFFFu;

// The group size, named once so the stride below cannot drift from the
// [numthreads] that produced it.  Its VALUE is this port's choice; what makes
// the kernel correct is the stride derived from it, not agreement with the
// engine's 1024 (see the header).
static const uint CVXCORE_PLANE_BLOCK = 64u;

// P9.57: FROM THE ENGINE ENUM's numbering, never a remembered literal.
// tools/check_geom_type_constants.py reads PxGeometry.h and checks these.
static const uint EGT_SPHERE = 0u;
static const uint EGT_PLANE = 1u;
static const uint EGT_CAPSULE = 2u;
static const uint EGT_BOX = 3u;
static const uint EGT_CONVEXCORE = 4u;

// PxsContactManagerStatusFlag
static const uint FLAG_HAS_NO_TOUCH = 1u;
static const uint FLAG_HAS_TOUCH = 2u;
static const uint FLAG_TOUCH_KNOWN = 3u;

// PxgPatchAndContactCounters::OverflowError
static const uint OVERFLOW_CONTACT = 1u;
static const uint OVERFLOW_FORCE = 2u;
static const uint OVERFLOW_PATCH = 4u;

static const uint CONTACT_SIZE = 16u; // sizeof(PxContact)
static const uint FORCE_SIZE = 4u;    // sizeof(PxU32)
static const uint PATCH_SIZE = 64u;   // sizeof(PxContactPatch)

// PxCombineMode / PxMaterialFlag
static const uint MODE_AVERAGE = 0u, MODE_MIN = 1u, MODE_MULTIPLY = 2u, MODE_MAX = 3u;
static const uint MAT_DISABLE_FRICTION = 1u, MAT_DISABLE_STRONG_FRICTION = 2u, MAT_COMPLIANT_ACC_SPRING = 16u;

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

// PxsCombineMaterials scalar combine (npInsertPatch combineMaterials).
float combineScalars(float a, float b, uint mode)
{
    if (mode == MODE_AVERAGE) return 0.5f * (a + b);
    if (mode == MODE_MIN) return min(a, b);
    if (mode == MODE_MULTIPLY) return a * b;
    if (mode == MODE_MAX) return max(a, b);
    return 0.0f;
}

// Gu::makeConvexShape (convexCoreCollision.cu:115-170), the four cases this
// kernel's slot 1 can present.  Returns false for a type it does not handle,
// exactly as the .cu's trailing `return false` does.
bool makeConvexShape(PxgShapeGpu s, float4 poseQ, float3 poseP,
                     out ShapeRef sh, out float margin)
{
    sh.type = 0u;
    sh.core[0] = uint4(0u, 0u, 0u, 0u);
    sh.core[1] = uint4(0u, 0u, 0u, 0u);
    sh.core[2] = uint4(0u, 0u, 0u, 0u);
    sh.poseQ = poseQ;
    sh.poseP = poseP;
    margin = 0.0f;

    if (s.type == EGT_CONVEXCORE)
    {
        // THE CORE TYPE IS THE POINTER FIELD -- see the header.
        sh.type = uint(s.hullOrMeshPtr);
        // memcpy(coreData, &scale.scale.x, MAX_CORE_SIZE): the core parameters
        // are the PxMeshScale's scale and rotation bytes, read as raw words.
        sh.core[0] = uint4(asuint(s.scaleX), asuint(s.scaleY), asuint(s.scaleZ),
                           asuint(s.scaleRotX));
        sh.core[1] = uint4(asuint(s.scaleRotY), asuint(s.scaleRotZ),
                           asuint(s.scaleRotW), 0u);
        margin = s.scaleRotW;
        return true;
    }
    if (s.type == EGT_SPHERE)
    {
        sh.type = CORE_POINT;
        margin = s.scaleX;
        return true;
    }
    if (s.type == EGT_CAPSULE)
    {
        sh.type = CORE_SEGMENT;
        sh.core[0].x = asuint(s.scaleX * 2.0f);   // SegmentCore::length
        margin = s.scaleY;
        return true;
    }
    if (s.type == EGT_BOX)
    {
        sh.type = CORE_BOX;
        sh.core[0] = uint4(asuint(s.scaleX * 2.0f), asuint(s.scaleY * 2.0f),
                           asuint(s.scaleZ * 2.0f), 0u);   // BoxCore::extents
        margin = 0.0f;
        return true;
    }
    return false;
}

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
    const bool ok = makeConvexShape(shape1, convexQ, convexP, sh, margin);

    // ---- Gu::generateContacts(PxPlane, ConvexShape) (GuConvexSupport.cpp:41-75)
    // Transcribed from npCvxCorePlaneGen's entry, which is the fixture that
    // verifies it.  `s.pts` there is the ePOINTS hull pool; this kernel's slot 1
    // is never eCONVEXMESH, so the pool is a zero stand-in and supportLocalQ's
    // CORE_POINTS branch is unreachable here.
    float4 ptsPool[12];
    for (uint z = 0u; z < 12u; ++z) ptsPool[z] = float4(0, 0, 0, 0);

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
    Ptr<PxgPatchAndContactCountersGpu> counters =
        Ptr<PxgPatchAndContactCountersGpu>(gParams.patchAndContactCounters);
    Ptr<PxsContactManagerOutputGpu> cmOutputs =
        Ptr<PxsContactManagerOutputGpu>(gParams.cmOutputs);
    Ptr<uint> touchChangeFlags = Ptr<uint>(gParams.touchChangeFlags);
    Ptr<uint> patchChangeFlags = Ptr<uint>(gParams.patchChangeFlags);

    uint contactByteOffset = INVALID;
    uint forceByteOffset = INVALID;
    if (nbContacts != 0u)
    {
        InterlockedAdd(counters[0].contactsBytes, CONTACT_SIZE * nbContacts, contactByteOffset);
        InterlockedAdd(counters[0].forceAndIndiceBytes, FORCE_SIZE * nbContacts, forceByteOffset);
        if ((contactByteOffset + CONTACT_SIZE) > gParams.contactBytesLimit)
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_CONTACT);
            contactByteOffset = INVALID;
        }
        if ((forceByteOffset + FORCE_SIZE) > gParams.forceBytesLimit)
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_FORCE);
            forceByteOffset = INVALID;
        }
    }

    if (contactByteOffset != INVALID)
    {
        // The points are ALREADY in world space here -- generateContacts works on
        // the world-space plane and the posed convex -- so unlike the convex-plane
        // kernel's tail there is no transform on the way out.
        Ptr<float4> baseContactStream = Ptr<float4>(gParams.contactStream + uint64_t(contactByteOffset));
        for (uint i = 0u; i < 4u; ++i)
            if (i < nbContacts)
                baseContactStream[i] = float4(pointsOut[i], distsOut[i]);

        cmOutputs[testIndex].contactForces = gParams.startContactForces + uint64_t(forceByteOffset);
        cmOutputs[testIndex].contactPoints = gParams.startContactPoints + uint64_t(contactByteOffset);
    }
    else
    {
        cmOutputs[testIndex].contactForces = 0uL;
        cmOutputs[testIndex].contactPoints = 0uL;
    }

    const uint allflags = cmOutputs[testIndex].allflagsWord;
    const uint oldStatusFlags = (allflags >> 16u) & 0xFFu;
    uint statusFlags = oldStatusFlags & (~FLAG_TOUCH_KNOWN);
    if (nbContacts != 0u) statusFlags |= FLAG_HAS_TOUCH;
    else                  statusFlags |= FLAG_HAS_NO_TOUCH;

    const uint prevPatches = (allflags >> 8u) & 0xFFu;
    const bool previouslyHadTouch = (oldStatusFlags & FLAG_HAS_TOUCH) != 0u;
    const bool prevTouchKnown = (oldStatusFlags & FLAG_TOUCH_KNOWN) != 0u;
    const uint numPatches = (nbContacts > 0u) ? 1u : 0u;
    const bool currentlyHasTouch = nbContacts > 0u;
    const bool change = (previouslyHadTouch != currentlyHasTouch) || (!prevTouchKnown);

    touchChangeFlags[testIndex] = change ? 1u : 0u;
    patchChangeFlags[testIndex] = (prevPatches != numPatches) ? 1u : 0u;

    cmOutputs[testIndex].allflagsWord = (prevPatches << 24u) | (statusFlags << 16u) | (numPatches << 8u);
    {
        const uint nf = cmOutputs[testIndex].nbContactsFlags;
        cmOutputs[testIndex].nbContactsFlags = (nf & 0xFFFF0000u) | (nbContacts & 0xFFFFu);
    }

    uint patchIndex = INVALID;
    if (nbContacts != 0u)
    {
        InterlockedAdd(counters[0].patchesBytes, PATCH_SIZE, patchIndex);
        if ((patchIndex + PATCH_SIZE) > gParams.patchBytesLimit)
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_PATCH);
            patchIndex = INVALID;

            statusFlags = (statusFlags & (~FLAG_TOUCH_KNOWN)) | FLAG_HAS_NO_TOUCH;
            cmOutputs[testIndex].allflagsWord = (prevPatches << 24u) | (statusFlags << 16u);
            const uint nf = cmOutputs[testIndex].nbContactsFlags;
            cmOutputs[testIndex].nbContactsFlags = (nf & 0xFFFF0000u);
            touchChangeFlags[testIndex] = previouslyHadTouch ? 1u : 0u;
            patchChangeFlags[testIndex] = (prevPatches != 0u) ? 1u : 0u;
        }
        else
        {
            cmOutputs[testIndex].contactPatches = gParams.startContactPatches + uint64_t(patchIndex);
        }
    }

    // insertIntoPatchStream(materials, patchStream, shape0, shape1, patchIndex,
    //   normal, nbContacts) -- inlined, as npCvxPlaneNphase inlines it.
    if (patchIndex != INVALID)
    {
        Ptr<PxsMaterialDataGpu> materials = Ptr<PxsMaterialDataGpu>(gParams.materials);
        const uint mi0 = shape0.materialIndex;
        const uint mi1 = shape1.materialIndex;
        const PxsMaterialDataGpu m0 = materials[mi0];
        const PxsMaterialDataGpu m1 = materials[mi1];

        const float r0f = m0.restitution, r1f = m1.restitution;
        const bool compliant0 = r0f < 0.0f, compliant1 = r1f < 0.0f;
        const bool exactlyOneCompliant = compliant0 != compliant1;
        const bool bothCompliant = compliant0 && compliant1;
        const uint m0flags = m0.flagsModes0 & 0xFFFFu;
        const uint m1flags = m1.flagsModes0 & 0xFFFFu;
        const uint m0fric = (m0.flagsModes0 >> 16u) & 0xFFu;
        const uint m1fric = (m1.flagsModes0 >> 16u) & 0xFFu;
        const uint m0rest = (m0.flagsModes0 >> 24u) & 0xFFu;
        const uint m1rest = (m1.flagsModes0 >> 24u) & 0xFFu;
        const uint m0damp = m0.modes1 & 0xFFu;
        const uint m1damp = m1.modes1 & 0xFFu;
        const bool cAcc0 = (m0flags & MAT_COMPLIANT_ACC_SPRING) != 0u;
        const bool cAcc1 = (m1flags & MAT_COMPLIANT_ACC_SPRING) != 0u;
        const bool exactlyOneAcc = cAcc0 != cAcc1;

        float rest, damping;
        if (bothCompliant && exactlyOneAcc)
            rest = cAcc0 ? r0f : r1f;
        else
        {
            const uint mode = exactlyOneCompliant ? MODE_MIN : max(m0rest, m1rest);
            const float flip = (bothCompliant && (mode == MODE_MULTIPLY)) ? -1.0f : 1.0f;
            rest = flip * combineScalars(r0f, r1f, mode);
        }
        if (bothCompliant && exactlyOneAcc)
            damping = cAcc0 ? m0.damping : m1.damping;
        else
        {
            const uint mode = exactlyOneCompliant ? MODE_MAX : max(m0damp, m1damp);
            damping = combineScalars(m0.damping, m1.damping, mode);
        }

        const uint combineFlags = m0flags | m1flags;
        float sta, dyn;
        uint matFlags;
        if ((combineFlags & MAT_DISABLE_FRICTION) == 0u)
        {
            const uint fm = max(m0fric, m1fric);
            const float dynF = combineScalars(m0.dynamicFriction, m1.dynamicFriction, fm);
            const float staF = combineScalars(m0.staticFriction, m1.staticFriction, fm);
            const float fDyn = max(dynF, 0.0f);
            dyn = fDyn;
            sta = (staF - fDyn) >= 0.0f ? staF : fDyn;
            matFlags = combineFlags;
        }
        else
        {
            matFlags = combineFlags | MAT_DISABLE_STRONG_FRICTION;
            dyn = 0.0f; sta = 0.0f;
        }

        Ptr<uint> patchBase = Ptr<uint>(gParams.patchStream + uint64_t(patchIndex));
        const uint one = asuint(1.0f);
        patchBase[0] = one; patchBase[1] = one; patchBase[2] = one; patchBase[3] = one;
        patchBase[4] = asuint(patchNormal.x);
        patchBase[5] = asuint(patchNormal.y);
        patchBase[6] = asuint(patchNormal.z);
        patchBase[7] = asuint(rest);
        patchBase[8] = asuint(dyn);
        patchBase[9] = asuint(sta);
        patchBase[10] = asuint(damping);
        patchBase[11] = ((nbContacts & 0xFFu) << 16u) | ((matFlags & 0xFFu) << 24u);
        patchBase[12] = (mi0 & 0xFFFFu) << 16u;
        patchBase[13] = (mi1 & 0xFFFFu);
    }
}

// The engine's block is 1024 and this group is CVXCORE_PLANE_BLOCK, so the loop
// runs 1024/BLOCK times per test batch; at the sizes the scenes reach it runs
// once and costs a compare.  `threadIndex / 1; // 1 tread per test` (:470-471) --
// upstream's typo kept -- is why the index is the flat thread id and not a warp.
[shader("compute")]
[numthreads(CVXCORE_PLANE_BLOCK, 1, 1)]
void convexCorePlaneNphase_Kernel(uint3 dtid : SV_DispatchThreadID)
{
    const uint stride = WorkgroupCount().x * CVXCORE_PLANE_BLOCK;
    for (uint testIndex = dtid.x; testIndex < gParams.numTests; testIndex += stride)
        convexCorePlaneOneTest(testIndex);
}
