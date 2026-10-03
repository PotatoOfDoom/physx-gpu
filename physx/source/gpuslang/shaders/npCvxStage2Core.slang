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

// npCvxStage2Core.slang — shared core for the two convexConvexNphase_stage2
// entries (npCvxStage2Setup = production 56-byte ABI, npCvxStage2SetupDbg =
// harness debug 64-byte blob).  Split into a module so each compiled entry
// declares exactly ONE push-constant block (two push_constant blocks in one
// module collide at offset 0).  See npCvxStage2Setup.slang for the full
// derivation.  ALL convex-convex type0 arms (sphere / capsule / convex hull)
// run the assembled npCvxCollide.collide(), gated only on transform validity
// exactly like the .cu (L2616).

module npCvxStage2Core;

import pxgSlangNpAbi;
import npCvxCollide;

public static const uint EGT_SPHERE = 0u;
public static const uint EGT_CAPSULE = 2u;
public static const uint EGT_BOX = 3u;

// Debug record mirroring the bound-buffer port's Stage2Out (the ss_scratch
// state the .cu populates), 13 x 16 B = 208 B per test.
public struct Stage2Debug
{
    public float4 rotOut0, rotOut1; // post-flip mesh-scale rotations
    public float4 tBq, tBp;         // scratch tB = post-flip transf1
    public float4 aToBq;
    public float4 aToBp;            // .w = contactDistance (with expandedMargin)
    public float4 verts0[2];        // synthetic sphere/capsule vertices
    public uint4 counts0;           // nbVertices0/nbPolygons0/nbEdges0/type0 (post-flip)
    public uint4 counts1;
    public float4 params;           // inSphereRadius0/1, replaceBreakingThreshold, expandedMargin
    public uint4 result;            // refreshedNb, initialNb, mNbContacts writeback word, flip
    public uint4 materials;         // shape0/shape1 material index (post-flip), transformsValid gate
}

public void stage2Body(uint globalWarpIndex, uint numTests, float toleranceLength,
                       uint64_t tempKeepPairsArray, uint64_t cmInputsPtr, uint64_t cmManifoldsPtr,
                       uint64_t shapesPtr, uint64_t transformCachePtr, uint64_t contactDistancePtr,
                       uint64_t debugOut)
{
    if (globalWarpIndex >= numTests)
        return;
    if ((Ptr<uint>(tempKeepPairsArray))[globalWarpIndex] == 0u)
        return;

    const PxgContactManagerInputGpu input =
        (Ptr<PxgContactManagerInputGpu>(cmInputsPtr))[globalWarpIndex];

    Ptr<PxgShapeGpu> shapes = Ptr<PxgShapeGpu>(shapesPtr);
    PxgShapeGpu shape0 = shapes[input.shapeRef0];
    PxgShapeGpu shape1 = shapes[input.shapeRef1];

    uint type0 = shape0.type;
    uint type1 = shape1.type;
    uint transformCacheRef0 = input.transformCacheRef0;
    uint transformCacheRef1 = input.transformCacheRef1;

    const bool flip = type1 < type0;
    if (flip)
    {
        const PxgShapeGpu ts = shape0; shape0 = shape1; shape1 = ts;
        const uint tr = transformCacheRef0; transformCacheRef0 = transformCacheRef1; transformCacheRef1 = tr;
    }
    type0 = shape0.type;
    type1 = shape1.type;

    Ptr<PxsCachedTransformGpu> cache = Ptr<PxsCachedTransformGpu>(transformCachePtr);
    const PxsCachedTransformGpu tc0 = cache[transformCacheRef0];
    const PxsCachedTransformGpu tc1 = cache[transformCacheRef1];
    const float4 t0q = float4(tc0.qx, tc0.qy, tc0.qz, tc0.qw);
    const float3 t0p = float3(tc0.px, tc0.py, tc0.pz);
    const float4 t1q = float4(tc1.qx, tc1.qy, tc1.qz, tc1.qw);
    const float3 t1p = float3(tc1.px, tc1.py, tc1.pz);

    // aToB = transf1.transformInv(transf0) (cudaGJKEPA.cu:2454), spelled as
    // PxTransform::transformInv(PxTransform) is (PxTransform.h:162-169): qinv =
    // q.getConjugate(); (qinv.rotate(src.p - p), qinv * src.q).  P9.94d: NOT
    // q.rotateInv(v) -- same value, but an exact-zero component can take the other sign.
    const float4 t1qInv = float4(-t1q.x, -t1q.y, -t1q.z, t1q.w);
    const float4 aToBq = npAbiQuatMul(t1qInv, t0q);
    const float3 aToBp = npAbiQuatRotate(t1qInv, t0p - t1p);

    Ptr<PxgPersistentContactManifoldGpu> manifold =
        Ptr<PxgPersistentContactManifoldGpu>(cmManifoldsPtr + uint64_t(globalWarpIndex) * 256u);

    const float3 scale0 = float3(shape0.scaleX, shape0.scaleY, shape0.scaleZ);
    const float3 scale1 = float3(shape1.scaleX, shape1.scaleY, shape1.scaleZ);
    const float4 rot0 = float4(shape0.scaleRotX, shape0.scaleRotY, shape0.scaleRotZ, shape0.scaleRotW);
    const float4 rot1 = float4(shape1.scaleRotX, shape1.scaleRotY, shape1.scaleRotZ, shape1.scaleRotW);

    // convexPtrA = hullPtr + sizeof(float4) — hoisted out of the convex branch
    // so collide() can consume it; never dereferenced for sphere/capsule
    // before collide (which itself only reads convexPtr1 on those arms).
    const uint64_t convexPtrA = shape0.hullOrMeshPtr + 16u;

    // synthetic vertices + margins (scratch-allocated verts in the .cu)
    float4 verts0_0 = float4(0, 0, 0, 0);
    float4 verts0_1 = float4(0, 0, 0, 0);
    float expandedMargin, convexMargin0;
    uint nbVertices0, nbPolygons0, nbEdges0;
    float inSphereRadius0;
    if (type0 == EGT_SPHERE)
    {
        verts0_0 = float4(0, 0, 0, 0);
        expandedMargin = scale0.x; // the CUDA reads .x here (not the .y used elsewhere)
        convexMargin0 = scale0.x;
        nbVertices0 = 1u; nbPolygons0 = 0u; nbEdges0 = 0u; inSphereRadius0 = 0.0f;
    }
    else if (type0 == EGT_CAPSULE)
    {
        verts0_0 = float4(1, 0, 0, 0);
        verts0_1 = float4(-1, 0, 0, 0);
        expandedMargin = scale0.y;
        convexMargin0 = scale0.y;
        nbVertices0 = 2u; nbPolygons0 = 0u; nbEdges0 = 0u; inSphereRadius0 = 0.0f;
    }
    else
    {
        // header uint4 at convexPtrA, extents at convexPtrA + sizeof(uint4)
        const float4 extents0 = (Ptr<float4>(convexPtrA + 16u))[0];
        const uint hdr0 = (Ptr<uint4>(convexPtrA))[0].x;
        expandedMargin = 0.0f;
        convexMargin0 = npAbiPcmConvexMargin(
            float3(extents0.x * scale0.x, extents0.y * scale0.y, extents0.z * scale0.z), toleranceLength);
        nbVertices0 = (hdr0 & 0xFFFFu) >> 8u; // getNbVerts
        nbPolygons0 = hdr0 & 0xFFu;           // getNbPolygons
        nbEdges0 = hdr0 >> 16u;               // getNbEdges
        // CUDA quirk ported verbatim: minScale0 is gated on SHAPE1's type (L2542)
        float minScale0 = 1.0f;
        if (type1 == EGT_BOX)
            minScale0 = min(scale0.z, min(scale0.x, scale0.y));
        inSphereRadius0 = extents0.w * minScale0;
    }

    const uint64_t convexPtrB = shape1.hullOrMeshPtr + 16u;
    const float4 extents1 = (Ptr<float4>(convexPtrB + 16u))[0];
    const uint hdr1 = (Ptr<uint4>(convexPtrB))[0].x;
    const uint nbVertices1 = (hdr1 & 0xFFFFu) >> 8u;
    const uint nbPolygons1 = hdr1 & 0xFFu;
    const uint nbEdges1 = hdr1 >> 16u;
    float minScale1 = 1.0f;
    if (type1 == EGT_BOX)
        minScale1 = min(scale1.z, min(scale1.x, scale1.y));
    const float inSphereRadius1 = extents1.w * minScale1;

    const float contactDist = (Ptr<float>(contactDistancePtr))[transformCacheRef0]
                            + (Ptr<float>(contactDistancePtr))[transformCacheRef1]
                            + expandedMargin;

    const float convexMargin1 = npAbiPcmConvexMargin(
        float3(extents1.x * scale1.x, extents1.y * scale1.y, extents1.z * scale1.z), toleranceLength);
    const float minMargin = min(convexMargin0, convexMargin1);
    const float replaceBreakingThreshold = minMargin * 0.05f;

    // the 16-lane moreData read; lane 12 holds the stage-1 counts word,
    // lanes 13-15 the warm-start words consumed by collide() (cu L2589-2592,
    // L1300-1302) — READ here, never stored back (cu epilogue stores 9..12).
    const float4 moreQuatB = manifold[0].quatB; // lanes 8..11
    const uint countsWord = manifold[0].nbContacts;
    const uint ws13 = manifold[0].nbWarmStartPoints;
    const uint ws14 = manifold[0].warmStartA;
    const uint ws15 = manifold[0].warmStartB;
    const uint initialNbContacts = countsWord >> 8u;
    uint numManifoldContacts = countsWord & 0xFFu;

    // collide() gate (cu L2615-2627): transform validity ONLY, matching the
    // .cu's `transf0.isValid() && transf1.isValid()`.  All three type0 arms
    // (SPHERE / CAPSULE / convex-hull generateConvexContacts) are assembled
    // in npCvxCollide.collide(); invalid-transform pairs keep the writeback
    // refreshedNb | 0x80000000 (bIntersection false, count unchanged).
    const bool transformsValid = npAbiTransformIsValid(t0q, t0p) && npAbiTransformIsValid(t1q, t1p);
    bool bIntersection = false;
    if (transformsValid)
    {
        CollideScratch cs;
        for (uint i = 0u; i < 64u; ++i) { cs.vA[i] = float4(0, 0, 0, 0); cs.vB[i] = float4(0, 0, 0, 0); }
        // cachedData.Reset (cu L2416); the image is zeroed for determinism
        // (uninitialized shared memory in the .cu; slots 16-31 are never read
        // before being written)
        for (uint i = 0u; i < 32u; ++i) cs.cachedData.v[i] = 0.0f;
        cs.cachedData.size = 0u;
        cs.gjkOutput.direction = float3(0, 0, 0);
        cs.gjkOutput.closestPointA = float3(0, 0, 0);
        cs.gjkOutput.closestPointB = float3(0, 0, 0);
        cs.gjkOutput.closestPointDir = float3(0, 0, 0);
        cs.gjkOutput.lowerBound = 0.0f;
        cs.gjkOutput.upperBound = 0.0f;
        cs.gjkOutput.degenerate = false;
        cs.scale0 = scale0; cs.rot0 = rot0;
        cs.scale1 = scale1; cs.rot1 = rot1;
        cs.aToBq = aToBq; cs.aToBp = aToBp;
        cs.manifoldBase = cmManifoldsPtr + uint64_t(globalWarpIndex) * 256u;
        cs.convexPtr0 = convexPtrA; cs.convexPtr1 = convexPtrB;
        cs.nbVertices0 = nbVertices0; cs.nbPolygons0 = nbPolygons0; cs.nbEdges0 = nbEdges0;
        cs.nbVertices1 = nbVertices1; cs.nbPolygons1 = nbPolygons1; cs.nbEdges1 = nbEdges1;
        cs.contactDistance = contactDist;
        cs.toleranceLength = toleranceLength;
        cs.replaceBreakingThreshold = replaceBreakingThreshold;
        cs.inSphereRadius0 = inSphereRadius0; cs.inSphereRadius1 = inSphereRadius1;
        cs.initialNbContacts = initialNbContacts;
        cs.type0 = type0; cs.type1 = type1;

        float4 verts0Synth[2] = { verts0_0, verts0_1 };
        // verts0Ptr = 0 selects the synthetic sphere/capsule vertices; the
        // convex-hull type0 reads the blob cloud at convexPtrA + sizeof(uint4)
        // + sizeof(float4) (cu L2485-2487), like hull B (cu L2498)
        const uint64_t verts0Ptr =
            (type0 == EGT_SPHERE || type0 == EGT_CAPSULE) ? 0 : convexPtrA + 32u;
        bIntersection = collide(cs, verts0Synth, verts0Ptr, convexPtrB + 32u,
                                ws13, ws14, ws15, numManifoldContacts);
    }

    // epilogue store: lanes 9..12 write moreData words 9..12
    // (quatB.y/z/w rewritten with the values read; word 12 = writeback)
    const uint writeback = numManifoldContacts | (bIntersection ? 0u : 0x80000000u);
    manifold[0].quatB = float4(moreQuatB.x, moreQuatB.y, moreQuatB.z, moreQuatB.w);
    manifold[0].nbContacts = writeback;

    if (debugOut != 0)
    {
        // index-based so Slang computes the physical stride (208 B); a manual
        // byte stride is a latent OOB if it ever drifts from the struct size
        Ptr<Stage2Debug> dbg = Ptr<Stage2Debug>(debugOut);
        dbg[globalWarpIndex].rotOut0 = rot0;
        dbg[globalWarpIndex].rotOut1 = rot1;
        dbg[globalWarpIndex].tBq = t1q;
        dbg[globalWarpIndex].tBp = float4(t1p, 0.0f);
        dbg[globalWarpIndex].aToBq = aToBq;
        dbg[globalWarpIndex].aToBp = float4(aToBp, contactDist);
        dbg[globalWarpIndex].verts0[0] = verts0_0;
        dbg[globalWarpIndex].verts0[1] = verts0_1;
        dbg[globalWarpIndex].counts0 = uint4(nbVertices0, nbPolygons0, nbEdges0, type0);
        dbg[globalWarpIndex].counts1 = uint4(nbVertices1, nbPolygons1, nbEdges1, type1);
        dbg[globalWarpIndex].params = float4(inSphereRadius0, inSphereRadius1, replaceBreakingThreshold, expandedMargin);
        dbg[globalWarpIndex].result = uint4(countsWord & 0xFFu, initialNbContacts, writeback, flip ? 1u : 0u);
        dbg[globalWarpIndex].materials = uint4(shape0.materialIndex, shape1.materialIndex, transformsValid ? 1u : 0u, 0u);
    }
}
