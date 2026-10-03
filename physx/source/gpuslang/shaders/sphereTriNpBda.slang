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

// sphereTriNpBda.slang -- `sphereTrimeshNarrowphase` (convexMesh.cu:762-975) on
// the engine BDA ABI.  P9.66b.
//
// The kernel that stops a sphere or a capsule CRASHING the simulation when it
// touches a cooked triangle mesh.  Before this, the launch was refused
// (`GPU sphereTrimeshCore fail to launch kernel!!`, PxgNarrowphaseCore.cpp:2027)
// and the process then died with SIGSEGV at the first step -- measured in P9.66a
// on PXG_SCENE=spheremesh and PXG_SCENE=capsulemesh, both exit 139.  The refused
// kernel leaves the contact-manager output uninitialised and a downstream consumer
// dereferences it, the same shape as the P9.52b-6 crash.
//
// ===========================================================================
// ONE KERNEL, TWO ORDINARY GEOMETRY PAIRS
// ===========================================================================
//
// PxgNphaseImplementationContext.cpp routes BOTH sphere-vs-mesh and
// capsule-vs-mesh here: `cndCaseSphereMesh` (:1046-1047) and `cndCaseCapsuleMesh`
// (:1070-1071) each end with
//
//     if(p.isMeshGPUCompatible1())
//         return GPU_BUCKET_ID::eSphereTrimesh;
//
// so one port fixes two pairs.  The two arms are NOT the same code -- the sphere
// arm is point-vs-triangle, the capsule arm segment-vs-triangle -- which is why
// P9.66a added two scenes rather than one.  A port that gets only the sphere arm
// right passes a spheremesh-only sweep.
//
// ===========================================================================
// THE INDEX ASYMMETRY: THE SPHERE ARM REMAPS, THE CAPSULE ARM DOES NOT
// ===========================================================================
//
// This is the defect this kernel is most likely to ship with, because it is
// SILENT and because the module one would naturally reuse gets it wrong.
//
//     convexMesh.cu:392  (sphere)   ... + remapCpuTriangleIdx;
//     convexMesh.cu:613  (capsule)  ... + triangleIdx;
//
// The sphere arm is the only one of the two that reads `trimeshFaceRemap` at all:
// it calls the FIVE-argument `readTriangleMesh` (triangleMesh.cuh:108-118), the
// capsule arm the four-argument overload.  `pxgSlangSphereHfCore` -- the
// structural twin, and the obvious thing to reach for -- packs the RAW index at
// its :427-428, and correctly so, because a heightfield has no face-remap table
// at all.  Reusing it here would be wrong by construction.
//
// Nothing catches this at runtime.  `getTriangleIndex()` (convexNpCommon.h:90-93)
// masks to 27 bits, so both candidates are small non-negative integers: no fault,
// no bounds check, no differing counter.  Contact positions, normals and
// separations are all identical.  Only the reported face identity is wrong.
//
// AND THE SCENE CANNOT CATCH IT EITHER.  `addMeshFloor` cooks a two-triangle quad,
// whose face remap is the identity, so `remapCpuTriangleIdx == triangleIdx` and
// the two spellings agree.  The fixture is where this has to be tested, on a mesh
// whose remap is provably non-identity -- see tests/testSphereTriNp.cpp.
//
// ===========================================================================
// THE GRID-STRIDE LOOP IS MANDATORY, AND FOR A SHARPER REASON THAN USUAL
// ===========================================================================
//
// The host launch (PxgNarrowphaseCore.cpp:1989-1990) is
//
//     const PxU32 numThreadsPerBlock = 64;
//     PxU32 numBlocks = (numTests + numThreadsPerBlock - 1) / numThreadsPerBlock;
//
// where `numTests` is the CONTACT MANAGER count.  The kernel does not loop over
// contact managers -- it loops over midphase PAIRS (`nbPairs`, a device-side
// value read from `nbPairsGlobal`), which is a different and much larger number.
// A two-body scene gives one block, 64 threads, and a mesh of any size gives far
// more than 64 sphere-triangle pairs.  So unlike the convex heightfield core --
// whose host over-provisions the grid 32-fold on purpose (`coreGridMultiplier`,
// :2898) -- there is no slack here at all, and a single-pass port would leave
// almost every pair's record unwritten.  convexMesh.cu:831:
//
//     for (PxU32 i = 0; i < nbPairs; i += gridDim.x * blockDim.x)
//
// The trip count is grid-uniform (the bound is on `i`, not on the per-lane
// `globalThreadIndex`), which is what makes the unconditional warp primitives in
// the output stage legal -- every lane runs the same number of iterations.
//
// ===========================================================================
// THE SLOT SWAP, WHICH THE HEIGHTFIELD SIBLING DOES NOT HAVE
// ===========================================================================
//
// convexMesh.cu:845-857 tests slot 0's type and swaps when it is the MESH:
//
//     bool flip = type0 == PxGeometryType::eTRIANGLEMESH;
//     if (flip) { PxSwap(shapeRef0, shapeRef1); PxSwap(transformCacheRef0, ...); ... }
//
// then re-reads `type0` from the swapped slot and dispatches on it.  The
// heightfield kernel reads slot 0 as the primitive unconditionally.  The swap is
// done HERE, in the entry, and the swapped refs are handed to the arms, because
// that is where the .cu does it and because the entry needs the post-swap type to
// choose an arm at all.
//
// ===========================================================================
// THE ABI ROW IS SHARED -- AND MUST BE EXACT
// ===========================================================================
//
// `sphereTrimeshNarrowphase` (:763-788) declares the SAME 21 parameters in the
// SAME order with the same types as `convexTrimeshNarrowphase` (:623-647): only
// the typedef spellings (`SphereMeshPair` / `ConvexMeshPair`, both typedefs of
// one struct at convexNpCommon.h:140-143) and one identifier differ.  So the blob
// is the same 168 bytes and argument 18 is the same PINNED HOST pointer, and this
// kernel's row points at `kCvxTriCore` with `kCvxTriCoreHostPtrMask`
// (PxgVulkanKernelAbi.cpp:1404-1405, :1432) like the other three sharers.
//
// P9.48d is why "close" is not good enough: `PxgCudaShim.cpp` requires an EXACT
// blob match, and a kernel whose ABI row is missing or wrong gets
// `pushConstantSizeFor() == 0` and every launch REFUSED IN SILENCE -- with
// fixtures, manifest test, suite and scenes all still green.
//
// ===========================================================================
// WHY THE ARMS ARE INLINE HERE AND NOT MODULES
// ===========================================================================
//
// The heightfield pair lives in `pxgSlangSphereHfCore` / `pxgSlangCapsuleHfCore`
// because TWO shaders call them -- the Phase-1 fixture and the BDA kernel.  Here
// the Phase-1 fixture (`npCvxMeshNp.slang`, entries cmSphereTri / cmCapsuleTri)
// already carries its own copies against bound buffers, and refactoring a passing
// fixture onto a new module boundary is a second change wearing the first one's
// clothes.  The heavy math IS shared, through the modules imported below; what is
// inline here is the BDA plumbing and the two arms' control flow.

#define PXG_WARP_BLOCK 32
#include "pxgSlangWarpInc.slang"

import pxgSlangCommon;            // quatBasisVector0
import pxgSlangMidphaseAlloc;     // maCompute
import pxgSlangCvxTriSetup;       // CtShape / CtContactManagerInput / the quat helpers
import pxgSlangPointTriangle;     // sphereTriangle.cuh:41, one copy (P9.56)
import pxgSlangSegTriangle;       // stSegmentTriangleSquared
import pxgSlangCapTriangle;       // capGenerateContacts / capGenerateEEContacts

// P9.57: FROM THE ENGINE ENUM's numbering, never a remembered literal.
// tools/check_geom_type_constants.py reads PxGeometry.h and checks these.
static const uint GEOM_SPHERE = 0u;
static const uint GEOM_CAPSULE = 2u;
static const uint GEOM_TRIANGLEMESH = 8u;

// PX_MAX_F32 -- what the CUDA seeds `separation` with (:296, :480) and therefore
// what lands in sphereTriMaxDepth for a pair that generates nothing.
static const float ST_MAX_F32 = 3.4028234663852886e+38f;
// FLT_EPSILON, the capsule arm's "segment intersects the triangle" test (:503).
static const float ST_FLT_EPSILON = 1.1920928955078125e-07f;

// ConvexTriNormalAndIndex (convexNpCommon.h:84-85), via the SphereTri typedef.
static const uint ST_DEFERRED_CONTACT_MASK = 0x80000000u;
static const uint ST_NB_CONTACTS_SHIFT = 27u;
// TriangleSearchFlags (convexTriangle.cuh:566-573).
static const uint ST_FACE_CONTACT_FLAG = 0x80000000u;
static const uint ST_EDGE_CONTACT_FLAG = 0xc0000000u;
// "around 5 degree" (:326, :515).
static const float ST_TOLERANCE = 0.996f;

// calculateConvexMeshPairMemRequirement() (midphaseAllocate.cuh:93).
static const uint ST_MEM_PER_PAIR = 56u;
// sizeof(ConvexTriContact) -- one float4.
static const uint ST_CONTACT_STRIDE = 16u;
// The packed BV32 node stride.  `pxgSlangCvxTriSetup` keeps its copy private, so
// this is a second one; both are the same number for the same reason and
// tools/check_cuda_twin_kernels.py is what would notice them drifting apart.
static const uint ST_BV32_STRIDE = 1168u;

// The 21-argument pack of convexMesh.cu:763-788 at CUDA's natural-alignment
// .param layout -- the same 168-byte blob as CvxTriCoreParams, field for field.
struct SphereTriNpParams
{
    uint64_t cmInputs;                 // @0
    uint64_t transformCache;           // @8
    uint64_t contactDistance;          // @16
    uint64_t gpuShapes;                // @24
    uint64_t sphereTrimeshPair;        // @32
    uint64_t sphereTriNIPtr;           // @40
    uint64_t sphereTriContactsPtr;     // @48
    uint64_t sphereTriMaxDepthPtr;     // @56
    uint64_t sphereTriIntermPtr;       // @64
    uint64_t orderedSphereTriIntermPtr;// @72
    uint64_t sphereTriSecondPassPairsPtr; // @80
    uint64_t stackPtr;                 // @88
    uint64_t nbPairsGlobal;            // @96
    uint64_t nbPaddedPairsGlobal;      // @104
    uint64_t nbSecondPassPairs;        // @112
    uint     stackSizeBytes;           // @120  (read by neither arm -- see below)
    uint     _pad124;                  // @124 (the next argument is 8-aligned)
    uint64_t tempConvexTriContacts;    // @128
    uint64_t pTempContactIndex;        // @136
    uint64_t maxTempMemRequirement;    // @144  PINNED HOST memory
    uint64_t midphasePairsNeeded;      // @152
    uint     nbContactManagers;        // @160
    uint     _pad164;                  // @164 (blob rounded to maxAlign 8)
}                                      // = 168

[[vk::push_constant]]
ConstantBuffer<SphereTriNpParams> gStn;

// ---------------------------------------------------------------------------
// the cooked mesh blob
// ---------------------------------------------------------------------------
// Same walk as `ctSetup` (pxgSlangCvxTriSetup.slang:355-365), reproduced rather
// than called because ctSetup transforms the triangle into CONVEX space and both
// arms here need it in MESH space.  The header layout is
// (numVerts, numTris, ?, nbPackedNodes) -- .z is NOT the node count.
struct StMesh
{
    uint64_t verts;
    uint64_t indices;
    uint64_t adj;
    uint64_t faceRemap;
}

StMesh stReadMesh(uint64_t meshPtr)
{
    const uint4 h = Ptr<uint4>(meshPtr)[0];
    StMesh m;
    m.verts = meshPtr + 16UL + uint64_t(h.w) * uint64_t(ST_BV32_STRIDE);
    m.indices = m.verts + uint64_t(h.x) * 16UL;
    m.adj = m.indices + uint64_t(h.y) * 16UL;
    m.faceRemap = m.adj + uint64_t(h.y) * 16UL;
    return m;
}

// getShapeSpaceVerts (convexMesh.cu:55-72) -> MeshScaling::getShapeSpaceVert
// (cudaNpCommon.h:241-259).  P9.94d: the IDENTITY-SCALE BYPASS is part of it.
// `idtMeshScale` is `scale == (1,1,1)` exactly (cudaNpCommon.h:197) and looks at
// the SCALE ONLY -- a non-identity scale rotation with a unit scale still takes
// the bypass, and the raw vertices go through untouched.  vertex2Shape there
// would be the identity in exact arithmetic, but `rotateInv(rot, rotate(rot, v))`
// rounds, so skipping it is observable (sphereTriNp_p94dIdtScaleRot).  No flip in
// that branch: a unit scale's determinant is +1.  Otherwise vertex2Shape per
// vertex, then the winding flip when the scale determinant is negative: lanes 1
// and 2 go to swapped slots; lane 0 never moves.  Same shape as the Phase-1 twin
// (npCvxMeshNp.slang getShapeSpaceVerts).
void stShapeSpaceVerts(StMesh m, uint triangleIdx, float3 scale, float4 scaleRot,
                       out float3 v0, out float3 v1, out float3 v2)
{
    const uint4 vi = Ptr<uint4>(m.indices)[triangleIdx];
    const float3 a = Ptr<float4>(m.verts)[vi.x].xyz;
    const float3 b = Ptr<float4>(m.verts)[vi.y].xyz;
    const float3 c = Ptr<float4>(m.verts)[vi.z].xyz;
    const bool idtMeshScale = (scale.x == 1.0f && scale.y == 1.0f && scale.z == 1.0f);
    if (idtMeshScale)
    {
        v0 = a;
        v1 = b;
        v2 = c;
    }
    else
    {
        const float3 sa = ctVertex2Shape(a, scale, scaleRot);
        const float3 sb = ctVertex2Shape(b, scale, scaleRot);
        const float3 sc = ctVertex2Shape(c, scale, scaleRot);
        const bool flipNormal = (scale.x * scale.y * scale.z) < 0.0f;
        v0 = sa;
        v1 = flipNormal ? sc : sb;
        v2 = flipNormal ? sb : sc;
    }
}

// getTriangleActiveMaskFromNormal (convexTriangle.cuh).  Only the CAPSULE arm
// calls it; the sphere arm gets its mask for free out of
// distancePointTriangleSquared.  Transcribed from the Phase-1 port at
// npCvxMeshNp.slang:360, which the tranche-68 fixture covers.
uint stActiveMaskFromNormal(float3 v0, float3 v1, float3 v2, float3 n)
{
    const float3 e0 = v1 - v0;
    const float3 e1 = v2 - v1;
    const float3 e2 = v0 - v2;
    const float d0 = ptDot(ptCross(e0, n), n);
    const float d1 = ptDot(ptCross(e1, n), n);
    const float d2 = ptDot(ptCross(e2, n), n);
    // the Voronoi region encoding of pxgSlangPointTriangle (PT_EV0 .. PT_EE12)
    if (d0 >= 0.0f && d2 < 0.0f) return PT_EV0;
    if (d1 >= 0.0f && d0 < 0.0f) return PT_EV1;
    if (d2 >= 0.0f && d1 < 0.0f) return PT_EV2;
    if (d0 < 0.0f && d1 < 0.0f) return PT_EE01;
    if (d1 < 0.0f && d2 < 0.0f) return PT_EE12;
    return PT_EE02;
}

// isEdgeContact (convexNpCommon.h): bit 31 of the region mask.
bool stIsEdgeContact(uint mask) { return (mask & 0x80000000u) != 0u; }

// ---------------------------------------------------------------------------
// what an arm produces
// ---------------------------------------------------------------------------
struct StArmOut
{
    uint nbOutputContacts;
    uint outMask;
    float3 contact0;
    float3 contact1;
    float sep0;
    float sep1;
    uint pairOffset;
}

StArmOut stArmInit()
{
    StArmOut o;
    o.nbOutputContacts = 0u;
    o.outMask = 0u;
    o.contact0 = float3(0.0f, 0.0f, 0.0f);
    o.contact1 = float3(0.0f, 0.0f, 0.0f);
    o.sep0 = 0.0f;
    o.sep1 = 0.0f;
    o.pairOffset = 0u;
    return o;
}

// ---------------------------------------------------------------------------
// the SPHERE arm -- sphereTrimeshNarrowphaseCore, convexMesh.cu:219-397
// ---------------------------------------------------------------------------
StArmOut stSphereArm(uint globalThreadIndex,
                     uint sphereShapeRef, uint meshShapeRef,
                     uint sphereXfRef, uint meshXfRef,
                     uint64_t ni, uint64_t maxDepth, uint64_t interm, uint64_t ordered,
                     uint4 curPair)
{
    StArmOut o = stArmInit();

    const uint cmIdx = curPair.x;
    const uint triangleIdx = curPair.y;
    const uint testOffset = curPair.z;

    Ptr<CtShape> shapes = Ptr<CtShape>(gStn.gpuShapes);
    const CtShape sphereShape = shapes[sphereShapeRef];
    const CtShape trimeshShape = shapes[meshShapeRef];

    // PxsCachedTransform, 32 B: PxQuat q @0, PxVec3 p @16.
    const uint64_t xf0 = gStn.transformCache + uint64_t(sphereXfRef) * 32UL;
    const uint64_t xf1 = gStn.transformCache + uint64_t(meshXfRef) * 32UL;
    const float4 sphereQ = Ptr<float4>(xf0)[0];
    const float3 sphereP = float3(Ptr<float>(xf0 + 16UL)[0], Ptr<float>(xf0 + 20UL)[0],
                                  Ptr<float>(xf0 + 24UL)[0]);
    const float4 meshQ = Ptr<float4>(xf1)[0];
    const float3 meshP = float3(Ptr<float>(xf1 + 16UL)[0], Ptr<float>(xf1 + 20UL)[0],
                                Ptr<float>(xf1 + 24UL)[0]);

    // ConvexMeshPair, 56 B: startIndex @28, roundedStartIndex @40.  The trailing
    // numbers in the .h declaration are RUNNING END offsets, not field offsets.
    const uint64_t pairAddr = gStn.sphereTrimeshPair + uint64_t(cmIdx) * 56UL;
    const uint pairOffset = Ptr<uint>(pairAddr + 28UL)[0] + testOffset;
    const uint pairOffsetPadded = Ptr<uint>(pairAddr + 40UL)[0] + testOffset;
    o.pairOffset = pairOffset;

    const float contactDist = Ptr<float>(gStn.contactDistance)[sphereXfRef]
                            + Ptr<float>(gStn.contactDistance)[meshXfRef];

    // PxSphereGeometry packs the radius into scale.x (PxgNarrowphaseCore.cpp:7508).
    const float sphereRadius = sphereShape.scaleX;
    const float3 trimeshScale = float3(trimeshShape.scaleX, trimeshShape.scaleY,
                                       trimeshShape.scaleZ);
    const float4 trimeshScaleRot = float4(trimeshShape.scaleRotX, trimeshShape.scaleRotY,
                                          trimeshShape.scaleRotZ, trimeshShape.scaleRotW);

    // trimeshTransform.transformInv(sphereTransform.p) -- spelled as the .cu
    // spells it (:262), not derived from a composed convex-relative transform.
    const float3 sphereCenterInTriMesh = ctQuatRotateInv(meshQ, sphereP - meshP);

    const uint64_t meshPtr = uint64_t(trimeshShape.hullOrMeshOffset.x)
                           | (uint64_t(trimeshShape.hullOrMeshOffset.y) << 32);
    const StMesh mesh = stReadMesh(meshPtr);

    float3 v0, v1, v2;
    stShapeSpaceVerts(mesh, triangleIdx, trimeshScale, trimeshScaleRot, v0, v1, v2);

    const float3 n = ptNormalized(ptCross(v1 - v0, v2 - v0));

    uint4 triAdj = Ptr<uint4>(mesh.adj)[triangleIdx];
    const bool flipNormal = (trimeshScale.x * trimeshScale.y * trimeshScale.z) < 0.0f;
    if (flipNormal)
    {
        const uint t = triAdj.x; triAdj.x = triAdj.z; triAdj.z = t;
    }

    // THE ASYMMETRY.  The sphere arm remaps; the capsule arm does not.  See header.
    const uint remapCpuTriangleIdx = Ptr<uint>(mesh.faceRemap)[triangleIdx];

    const float inflatedRadius = sphereRadius + contactDist;

    const float d = ptDot(v0, n);
    const float dist0 = ptDot(sphereCenterInTriMesh, n) - d;

    float separation = ST_MAX_F32;
    bool delayContacts = false;
    bool edgeContact = false;
    float3 patchNormal = float3(0.0f, 0.0f, 0.0f);

    if (dist0 >= 0.0f)
    {
        float3 closestP;
        bool generateContact;
        bool faceContact;
        uint mask;
        const float sqDist = ptDistancePointTriangleSquared(sphereCenterInTriMesh,
                                                           v0, v1, v2, triAdj,
                                                           closestP, generateContact,
                                                           faceContact, mask);

        const float sqInflatedSphereRadius = inflatedRadius * inflatedRadius;
        if (sqInflatedSphereRadius > sqDist && generateContact)
        {
            patchNormal = n;
            if (!faceContact)
                patchNormal = ptNormalized(sphereCenterInTriMesh - closestP);

            const float cosTheta = ptDot(patchNormal, n);
            delayContacts = cosTheta <= ST_TOLERANCE;
            if (delayContacts)
            {
                // `|` here and `+` on the NI index below -- upstream spells the
                // two differently over disjoint fields, and so does this port.
                o.outMask = mask | globalThreadIndex;
                edgeContact = stIsEdgeContact(mask);
            }

            o.nbOutputContacts = 1u;
            separation = sqrt(sqDist);

            o.sep0 = separation;
            // The sphere's contact is its own local origin (:360).  The radius is
            // NOT subtracted from the separation -- transcribed as written.
            o.contact0 = float3(0.0f, 0.0f, 0.0f);
        }
    }

    // ---- the per-pair records (:364-392) -----------------------------------
    Ptr<float>(maxDepth)[pairOffset] = separation;
    Ptr<uint>(interm)[pairOffset] = triangleIdx;

    // BOTH flags here, unlike the heightfield sphere core, which only ever writes
    // the FACE one because its capsule sibling has no deferral channel at all.
    uint orderedTriangleIdx = triangleIdx;
    if (o.nbOutputContacts != 0u)
    {
        if (!delayContacts)      orderedTriangleIdx |= ST_FACE_CONTACT_FLAG;
        else if (edgeContact)    orderedTriangleIdx |= ST_EDGE_CONTACT_FLAG;
    }
    // At the PADDED offset -- the sort's numbering, not the record numbering.
    Ptr<uint>(ordered)[pairOffsetPadded] = orderedTriangleIdx;

    const uint delayContactMask = delayContacts ? ST_DEFERRED_CONTACT_MASK : 0u;
    // Into the SPHERE's frame by way of the world (:387-388): rotate by the mesh,
    // rotate back by the sphere.  Both are rotations alone.
    const float3 worldNormal = ctQuatRotate(meshQ, -patchNormal);
    const float3 niNormal = ctQuatRotateInv(sphereQ, worldNormal);
    // THE REMAPPED index -- the one thing that distinguishes this arm's record
    // from the capsule arm's and from the heightfield core's.  See the header.
    const uint niIndex = delayContactMask + (o.nbOutputContacts << ST_NB_CONTACTS_SHIFT)
                       + remapCpuTriangleIdx;

    Ptr<uint>(ni)[4u * pairOffset + 0u] = asuint(niNormal.x);
    Ptr<uint>(ni)[4u * pairOffset + 1u] = asuint(niNormal.y);
    Ptr<uint>(ni)[4u * pairOffset + 2u] = asuint(niNormal.z);
    Ptr<uint>(ni)[4u * pairOffset + 3u] = niIndex;

    return o;
}

// ---------------------------------------------------------------------------
// the CAPSULE arm -- capsuleTrimeshNarrowphaseCore, convexMesh.cu:400-618
// ---------------------------------------------------------------------------
StArmOut stCapsuleArm(uint globalThreadIndex,
                      uint capsuleShapeRef, uint meshShapeRef,
                      uint capsuleXfRef, uint meshXfRef,
                      uint64_t ni, uint64_t maxDepth, uint64_t interm, uint64_t ordered,
                      uint4 curPair)
{
    StArmOut o = stArmInit();

    const uint cmIdx = curPair.x;
    const uint triangleIdx = curPair.y;
    const uint testOffset = curPair.z;

    Ptr<CtShape> shapes = Ptr<CtShape>(gStn.gpuShapes);
    const CtShape capsuleShape = shapes[capsuleShapeRef];
    const CtShape trimeshShape = shapes[meshShapeRef];

    const uint64_t xf0 = gStn.transformCache + uint64_t(capsuleXfRef) * 32UL;
    const uint64_t xf1 = gStn.transformCache + uint64_t(meshXfRef) * 32UL;
    const float4 capQ = Ptr<float4>(xf0)[0];
    const float3 capP = float3(Ptr<float>(xf0 + 16UL)[0], Ptr<float>(xf0 + 20UL)[0],
                               Ptr<float>(xf0 + 24UL)[0]);
    const float4 meshQ = Ptr<float4>(xf1)[0];
    const float3 meshP = float3(Ptr<float>(xf1 + 16UL)[0], Ptr<float>(xf1 + 20UL)[0],
                                Ptr<float>(xf1 + 24UL)[0]);

    const uint64_t pairAddr = gStn.sphereTrimeshPair + uint64_t(cmIdx) * 56UL;
    const uint pairOffset = Ptr<uint>(pairAddr + 28UL)[0] + testOffset;
    const uint pairOffsetPadded = Ptr<uint>(pairAddr + 40UL)[0] + testOffset;
    o.pairOffset = pairOffset;

    const float contactDist = Ptr<float>(gStn.contactDistance)[capsuleXfRef]
                            + Ptr<float>(gStn.contactDistance)[meshXfRef];

    // PxCapsuleGeometry packs (halfHeight, radius, radius) (PxgNarrowphaseCore.cpp:7515).
    const float capsuleRadius = capsuleShape.scaleY;
    const float capsuleHalfHeight = capsuleShape.scaleX;
    const float3 trimeshScale = float3(trimeshShape.scaleX, trimeshShape.scaleY,
                                       trimeshShape.scaleZ);
    const float4 trimeshScaleRot = float4(trimeshShape.scaleRotX, trimeshShape.scaleRotY,
                                          trimeshShape.scaleRotZ, trimeshShape.scaleRotW);

    // meshToCapsule = capsuleTransform.transformInv(trimeshTransform) (:443),
    // PxTransform.h:162-169: `qinv = q.getConjugate(); (qinv.rotate(src.p - p),
    // qinv * src.q)`.  P9.94d: rotate by the CONJUGATE, literally -- rotateInv(q)
    // matches it in every nonzero bit but can flip the sign of an exact-zero
    // component (the zero cross term is added in one, subtracted in the other),
    // and that sign reaches the stored contact (sphereTriNp_p94dCapsuleMtcZeroSign).
    const float4 capQinv = ctQuatConjugate(capQ);
    const float4 meshToCapQ = ctQuatMul(capQinv, meshQ);
    const float3 meshToCapP = ctQuatRotate(capQinv, meshP - capP);

    // getBasisVector0 is its OWN formula, not rotate(q,(1,0,0)) -- see
    // pxgSlangCapsuleHfCore's header.  Unlike the heightfield twin (upstream
    // candidate 50), this arm DOES rotate it into mesh space (:447).
    const float3 tmp = quatBasisVector0(capQ) * capsuleHalfHeight;
    const float3 capsuleCenterInMesh = ctQuatRotateInv(meshQ, capP - meshP);
    const float3 capsuleDirInMesh = ctQuatRotateInv(meshQ, tmp);
    const float3 p0 = capsuleCenterInMesh + capsuleDirInMesh;
    const float3 p1 = capsuleCenterInMesh - capsuleDirInMesh;

    const uint64_t meshPtr = uint64_t(trimeshShape.hullOrMeshOffset.x)
                           | (uint64_t(trimeshShape.hullOrMeshOffset.y) << 32);
    const StMesh mesh = stReadMesh(meshPtr);

    float3 v0, v1, v2;
    stShapeSpaceVerts(mesh, triangleIdx, trimeshScale, trimeshScaleRot, v0, v1, v2);

    const float3 triNormal = ptNormalized(ptCross(v1 - v0, v2 - v0));

    // Read and swapped exactly as the sphere arm does, and then never used: the
    // capsule arm's mask comes from the patch normal, not from adjacency.
    // Transcribed because the .cu reads it (:470-472).
    uint4 triAdj = Ptr<uint4>(mesh.adj)[triangleIdx];
    const bool flipNormal = (trimeshScale.x * trimeshScale.y * trimeshScale.z) < 0.0f;
    if (flipNormal)
    {
        const uint t = triAdj.x; triAdj.x = triAdj.z; triAdj.z = t;
    }

    const float inflatedRadius = capsuleRadius + contactDist;

    // NOTE the different spelling from the sphere arm's `dot(c,n) - dot(v0,n)`:
    // the capsule arm subtracts first (:479).  Same quantity, different rounding.
    const float dist = ptDot(capsuleCenterInMesh - v0, triNormal);

    float separation = ST_MAX_F32;
    bool delayContacts = false;
    bool edgeContact = false;
    float3 patchNormal = float3(0.0f, 0.0f, 0.0f);

    if (dist >= 0.0f)
    {
        float t, u, v;
        const float sqDist = stSegmentTriangleSquared(p0, p1, v0, v1, v2, t, u, v);
        const float sqInflatedRadius = inflatedRadius * inflatedRadius;

        if (sqInflatedRadius > sqDist)
        {
            // THE WHOLE cosTheta / delayContacts COMPUTATION IS INSIDE THE ELSE
            // (:508-521).  The sphere arm computes it unconditionally.  Hoisting
            // it here would change behaviour for every deeply penetrating capsule.
            if (sqDist < ST_FLT_EPSILON)
            {
                patchNormal = triNormal;
            }
            else
            {
                const float3 pointOnSegment = p0 + (p1 - p0) * t;
                const float3 pointOnTriangle = v0 * (1.0f - u - v) + v1 * u + v2 * v;
                patchNormal = ptNormalized(pointOnSegment - pointOnTriangle);

                const float cosTheta = ptDot(patchNormal, triNormal);
                delayContacts = cosTheta <= ST_TOLERANCE;
            }

            if (delayContacts)
            {
                const uint mask = stActiveMaskFromNormal(v0, v1, v2, patchNormal);
                o.outMask = mask | globalThreadIndex;
                edgeContact = stIsEdgeContact(mask);
            }

            CapContacts cc = capGenerateContacts(v0, v1, v2, triNormal, patchNormal,
                                                 p0, p1, inflatedRadius);
            cc = capGenerateEEContacts(v0, v1, v2, patchNormal, p0, p1,
                                       sqInflatedRadius, cc);

            o.nbOutputContacts = cc.n;
            // Contacts go back into CAPSULE space, and separation is the MINIMUM
            // over them (:570-576).
            for (uint i = 0u; i < 2u; ++i)
            {
                if (i < cc.n)
                {
                    const float3 c = ctQuatRotate(meshToCapQ, cc.pt[i]) + meshToCapP;
                    if (i == 0u) { o.contact0 = c; o.sep0 = cc.sep[i]; }
                    else         { o.contact1 = c; o.sep1 = cc.sep[i]; }
                    // P9.86: convexMesh.cu:575 PxMin is selectMin (a < b ? a : b), so a tie keeps pen: PxMin(-0,+0) = +0.  GLSL FMin on llvmpipe returns -0 in either order.
                    separation = stMin(separation, cc.sep[i]);
                }
            }
        }
    }

    // ---- the per-pair records (:580-613) -----------------------------------
    Ptr<uint>(interm)[pairOffset] = triangleIdx;

    uint orderedTriangleIdx = triangleIdx;
    if (o.nbOutputContacts != 0u)
    {
        if (!delayContacts)      orderedTriangleIdx |= ST_FACE_CONTACT_FLAG;
        else if (edgeContact)    orderedTriangleIdx |= ST_EDGE_CONTACT_FLAG;
    }
    Ptr<uint>(ordered)[pairOffsetPadded] = orderedTriangleIdx;

    Ptr<float>(maxDepth)[pairOffset] = separation;

    const uint delayContactMask = delayContacts ? ST_DEFERRED_CONTACT_MASK : 0u;
    // meshToCapsule.rotate(-patchNormal) (:611) -- ONE rotation, not the sphere
    // arm's world round trip.
    const float3 niNormal = ctQuatRotate(meshToCapQ, -patchNormal);
    // THE RAW index (:613).  The capsule arm never reads faceRemap -- it calls the
    // four-argument readTriangleMesh.  Faithful asymmetry, not an oversight.
    const uint niIndex = delayContactMask + (o.nbOutputContacts << ST_NB_CONTACTS_SHIFT)
                       + triangleIdx;

    Ptr<uint>(ni)[4u * pairOffset + 0u] = asuint(niNormal.x);
    Ptr<uint>(ni)[4u * pairOffset + 1u] = asuint(niNormal.y);
    Ptr<uint>(ni)[4u * pairOffset + 2u] = asuint(niNormal.z);
    Ptr<uint>(ni)[4u * pairOffset + 3u] = niIndex;

    return o;
}

// ---------------------------------------------------------------------------
// the entry -- sphereTrimeshNarrowphase, convexMesh.cu:762-975
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void sphereTrimeshNarrowphase(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    const uint nbPairs = Ptr<uint>(gStn.nbPairsGlobal)[0];
    const uint nbPaddedPairs = Ptr<uint>(gStn.nbPaddedPairsGlobal)[0];

    // Every CUDA block computes its own private copy (:806-816).
    const MidphaseAlloc a = maCompute(gStn.stackPtr, nbPairs, nbPaddedPairs);

    if (gid.x == 0u && lane == 0u)
    {
        // BY NAME, never by position: maCompute returns its seven addresses in
        // ALLOCATION order, which is not the order of the CUDA function's
        // out-parameters, and every one points inside the same stack allocation.
        Ptr<uint64_t>(gStn.sphereTriNIPtr)[0] = a.ni;
        Ptr<uint64_t>(gStn.sphereTriContactsPtr)[0] = a.contacts;
        Ptr<uint64_t>(gStn.sphereTriMaxDepthPtr)[0] = a.maxDepth;
        Ptr<uint64_t>(gStn.sphereTriIntermPtr)[0] = a.interm;
        Ptr<uint64_t>(gStn.orderedSphereTriIntermPtr)[0] = a.ordered;
        Ptr<uint64_t>(gStn.sphereTriSecondPassPairsPtr)[0] = a.secondPass;

        // The atomic lands in PINNED HOST memory (:812).
        const uint afterPairPadding = 3u * (ST_MEM_PER_PAIR - 2u * 4u);
        const uint afterTriangleIndexPadding = gStn.nbContactManagers * 6u * 4u;
        const uint required = ST_MEM_PER_PAIR * Ptr<uint>(gStn.midphasePairsNeeded)[0]
                            + afterPairPadding + afterTriangleIndexPadding;
        uint old;
        InterlockedMax(Ptr<uint>(gStn.maxTempMemRequirement)[0], required, old);
    }

    // convexMesh.cu:831.  See the header: this launch has NO grid multiplier, and
    // the grid is sized to the CONTACT MANAGER count while the loop runs over
    // PAIRS.  The trip count is workgroup-uniform, which is what keeps the
    // output stage's barriers in uniform control flow.
    const uint gridStride = WorkgroupCount().x * PXG_WARP_SIZE;

    for (uint base = 0u; base < nbPairs; base += gridStride)
    {
    const uint globalThreadIndex = gid.x * PXG_WARP_SIZE + lane + base;

    StArmOut arm = stArmInit();
    uint4 curPair = uint4(0u, 0u, 0u, 0u);

    if (globalThreadIndex < nbPairs)
    {
        curPair = Ptr<uint4>(a.pairs)[globalThreadIndex];

        const CtContactManagerInput cin =
            Ptr<CtContactManagerInput>(gStn.cmInputs)[curPair.x];

        // THE SWAP (:845-857), which the heightfield sibling does not have.
        uint shapeRef0 = cin.shapeRef0;
        uint shapeRef1 = cin.shapeRef1;
        uint xfRef0 = cin.transformCacheRef0;
        uint xfRef1 = cin.transformCacheRef1;

        Ptr<CtShape> shapes = Ptr<CtShape>(gStn.gpuShapes);
        if (shapes[shapeRef0].type == GEOM_TRIANGLEMESH)
        {
            const uint s = shapeRef0; shapeRef0 = shapeRef1; shapeRef1 = s;
            const uint x = xfRef0;    xfRef0 = xfRef1;       xfRef1 = x;
        }

        // Re-read the type from the POST-SWAP slot, as the .cu does (:856).
        const uint type0 = shapes[shapeRef0].type;

        if (type0 == GEOM_SPHERE)
        {
            arm = stSphereArm(globalThreadIndex, shapeRef0, shapeRef1, xfRef0, xfRef1,
                              a.ni, a.maxDepth, a.interm, a.ordered, curPair);
        }
        else if (type0 == GEOM_CAPSULE)
        {
            arm = stCapsuleArm(globalThreadIndex, shapeRef0, shapeRef1, xfRef0, xfRef1,
                               a.ni, a.maxDepth, a.interm, a.ordered, curPair);
        }
    }

    // ---- the output stage (:919-970), shared by both arms --------------------
    //
    // OUTSIDE the pair guard, exactly as the CUDA has it: every lane reaches every
    // primitive below whether or not it had a pair.  See sphereHfNpBda's header for
    // why the two broadcasts are hoisted out of their guards -- on this port's
    // default warp path a conditional shuffle is a conditional BARRIER.
    const uint nbContacts = arm.nbOutputContacts;
    const uint outMask = arm.outMask;

    const uint outMaskWriteMask = pxgBallotSync(PXG_FULL_MASK, lane, outMask != 0u);
    const uint count = pxgPopc(outMaskWriteMask);

    const uint inclusiveSum =
        pxgWarpScanAddInclusiveU(PXG_FULL_MASK, lane, nbContacts, 5u);
    const uint totalContacts = pxgShflSyncU(PXG_FULL_MASK, lane, inclusiveSum, 31, PXG_WARP_SIZE);

    uint secondPassStart = 0u;
    if (totalContacts != 0u && count != 0u && lane == 0u)
        InterlockedAdd(Ptr<uint>(gStn.nbSecondPassPairs)[0], count, secondPassStart);
    secondPassStart = pxgShflSyncU(PXG_FULL_MASK, lane, secondPassStart, 0, PXG_WARP_SIZE);

    uint contactStart = 0u;
    if (totalContacts != 0u && lane == 31u)
        InterlockedAdd(Ptr<uint>(gStn.pTempContactIndex)[0], inclusiveSum, contactStart);
    contactStart = pxgShflSyncU(PXG_FULL_MASK, lane, contactStart, 31, PXG_WARP_SIZE);

    if (totalContacts != 0u)
    {
        if (count != 0u && outMask != 0u)
        {
            const uint offset = pxgWarpRankInMask(outMaskWriteMask, lane);
            Ptr<uint>(a.secondPass)[secondPassStart + offset] = outMask;
        }

        // `inclusiveSum - nbContacts` is this lane's EXCLUSIVE prefix; the CUDA
        // derives it by subtraction rather than running a second scan (:965).
        contactStart += inclusiveSum - nbContacts;

        // NO BOUNDS TEST, and `stackSizeBytes` is passed in and never read on this
        // path -- transcribed as written.  A port that adds a guard its source
        // lacks stops being a port.
        for (uint i = 0u; i < 2u; ++i)
        {
            if (i < nbContacts)
            {
                const float3 c = (i == 0u) ? arm.contact0 : arm.contact1;
                const float s = (i == 0u) ? arm.sep0 : arm.sep1;
                const uint64_t dst = gStn.tempConvexTriContacts
                                   + uint64_t(contactStart + i) * uint64_t(ST_CONTACT_STRIDE);
                Ptr<float4>(dst)[0] = float4(c, s);
            }
        }

        if (nbContacts != 0u)
        {
            Ptr<uint>(a.contacts)[arm.pairOffset] = contactStart;
        }
    }
    }
}
