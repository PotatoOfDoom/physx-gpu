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

// npTriPlaneNp.slang
//
// Slang port of trimeshPlaneNarrowphase (trimeshCollision.cu :61-434),
// the block-level (32-warp) triangle-mesh-vs-plane narrowphase:
//
// 1. margin: the mesh extents come from a warp min/max reduction over the
//    BV32 ROOT node's child bounds (up to mNbNodes <= 32 slots);
//    minMargin = calculatePCMConvexMargin(extents, meshScale,
//    toleranceLength) * 0.5f with ratio 0.1 and breaking ratio 0.5;
// 2. the persistent-manifold gate (manifold.cuh, tranche-66
//    transcriptions): invalidateManifold on trimeshToPlane =
//    planeTransform.transformInv(trimeshTransform), then gated
//    refreshManifolds with per-contact penetration write-backs and
//    any-drift mNbManifolds zeroing;
// 3. FULL GEN: vertices are processed in block iterations of 1024 (32
//    warps x 32 lanes).  Per lane: p = vertex2Shape(vert), pInPlaneSpace
//    = trimeshToPlane.transform(p), INCLUSIVE accept `cDistance >=
//    pInPlaneSpace.x`, separation = the raw plane-space x, worldPoint =
//    trimeshTransform.transform(p).  Each warp runs
//    contactReduce<clustering=true, keepDeepestPCM=true, maxPoints=6,
//    dupeTest=false>(worldNormal = plane basis vector 0,
//    clusterTolerance) — the tranche-82 core; the per-warp survivors are
//    packed into a shared buffer via a block scan of the 32 warp counts
//    BEHIND the carried retainedCount, and warp 0 then re-reduces the
//    whole buffer with the readMask BATCHING loop: lanes without a
//    contact refill from the buffer (readIndex = exclusive-scan of the
//    free-lane mask + ind), `ind += popc(readMask)` advances by the
//    number of FREE lanes whether or not they read (faithful quirk: ind
//    can overshoot totalContacts), each round re-reducing the union of
//    kept + refilled points; the <= 6 survivors are compacted to the
//    buffer front and retainedCount carries into the next block
//    iteration;
// 4. PCM store: retained contacts write manifold 0 — pointA =
//    trimeshTransform.transformInv(world), pointB = the plane-space point
//    with x FORCED TO 0, normal = (1,0,0) in plane space, penetration =
//    the stored separation, triIndex = 0xFFFFFFFF — plus mNbManifolds =
//    1 (or 0 when nothing was retained) and mRelativeTransform =
//    trimeshToPlane;
// 5. CACHED path (no full gen): contacts reload from manifold 0 —
//    world = trimeshTransform.transform(pointA), separation = the cached
//    penetration (post-refresh, i.e. the freshly written-back values);
// 6. the contact-stream tail: ONE reservation pair on
//    contactsBytes/forceAndIndiceBytes (sizeof(PxContact) = 16 /
//    sizeof(PxU32) = 4 per contact) with overflow detection (flag set +
//    offset poisoned to 0xFFFFFFFF => the pair emits NOTHING and
//    nbContacts is zeroed), float4 (point, separation) writes, and the
//    output record's contactForces/contactPoints byte offsets.  The
//    registerContactPatch + insertIntoPatchStream tail is the
//    tranche-44/45 machinery composed unchanged (documented; this port
//    outputs nbContacts + worldNormal, the exact inputs that tail
//    consumes).
//
// WARP SERIALIZATION: one thread serializes the whole 32-warp block —
// warp ballots become lane arrays, the block scan a prefix sum, the
// cross-warp shared buffer a local array ((32+1)*6 slots), and the
// warp-0 batching loop a faithful serial transcription.  All decisions
// compare exactly-computed values (selections, IEEE-rounded sqrt on both
// sides), so the port is bit-deterministic against the reference.
//
// Engine glue not ported (documented): LoadShapePairWarp/transform
// caches, readTriangleMesh blob walk (cases carry vert ranges + the
// root-node bounds), grid distribution, and the patch-stream tail above.

import pxgSlangCommon;
// P9.82: the decision core -- vertex2Shape, calculatePCMConvexMargin, the arg-min/max
// helpers, contactReduce6, the batching loop and the two manifold predicates -- now lives
// in a module that the ENGINE kernel (triPlaneNpBda.slang) compiles too, so this fixture
// certifies the code the engine runs rather than a second transcription of it.  The
// bodies were MOVED, not retyped; what stayed here is buffer access, which does no
// arithmetic.
import pxgSlangTriPlane;

static const uint OVERFLOW_CONTACT = 1u;
static const uint OVERFLOW_FORCE = 2u;

struct TpParams
{
    uint numCases;
    uint contactBytesLimit;
    uint forceBytesLimit;
    uint pad0;
}

struct TpCase
{
    float4 meshQ;     // trimesh transform rotation
    float4 meshP;     // trimesh transform position; w = toleranceLength
    float4 planeQ;    // plane transform rotation
    float4 planeP;    // plane transform position; w = cDistance
    float4 scale;     // trimesh scale (xyz); w = clusterTolerance
    float4 scaleRot;  // trimesh scale rotation quaternion
    uint4 verts;      // x = vert offset, y = numVerts
    uint4 rootMeta;   // x = root mNbNodes, y = root bounds offset (slots of 2 float4)
}

[[vk::push_constant]]
ConstantBuffer<TpParams> gParams;

RWStructuredBuffer<TpCase> gCase;         // binding 0
RWStructuredBuffer<float4> gVerts;        // binding 1: mesh vertices (vertex space)
RWStructuredBuffer<float4> gRoot;         // binding 2: root child bounds, pairs (min, max) per slot
// manifold: 24 slots per case (pointA [w = triIndex bits], pointB, normalPen)
RWStructuredBuffer<float4> gManPointA;    // binding 3
RWStructuredBuffer<float4> gManPointB;    // binding 4
RWStructuredBuffer<float4> gManNPen;      // binding 5
RWStructuredBuffer<uint4> gManMeta;       // binding 6: per case (nbManifolds, nbContacts0, nbContacts1, nbContacts2) IN/OUT
RWStructuredBuffer<float4> gManRel;       // binding 7: per case 2 slots (q, p) IN/OUT
RWStructuredBuffer<uint> gCounters;       // binding 8: [0] contactsBytes, [1] forceAndIndiceBytes, [2] overflowError
RWStructuredBuffer<float4> gContactStream;// binding 9: byte offset / 16 indexing
RWStructuredBuffer<uint4> gOutRec;        // binding 10: per case (contactPointsOfs, contactForcesOfs, nbContacts, fullGen)
RWStructuredBuffer<float4> gDebug;        // binding 11: per case (worldNormal, retainedCount)

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void tpNarrowphase(uint3 dtid : SV_DispatchThreadID)
{
    const uint ci = dtid.x;
    if (ci >= gParams.numCases) return;
    const TpCase cs = gCase[ci];
    const float toleranceLength = cs.meshP.w;
    const float cDistance = cs.planeP.w;
    const float clusterTolerance = cs.scale.w;

    // 1. extents from the BV32 root node (min/max over <= 32 child slots)
    float3 mn = float3(FLT_MAX_LOCAL, FLT_MAX_LOCAL, FLT_MAX_LOCAL);
    float3 mx = float3(-FLT_MAX_LOCAL, -FLT_MAX_LOCAL, -FLT_MAX_LOCAL);
    for (uint s = 0; s < min(cs.rootMeta.x, 32u); ++s) // one warp in CUDA
    {
        const float3 bmn = gRoot[(cs.rootMeta.y + s) * 2 + 0].xyz;
        const float3 bmx = gRoot[(cs.rootMeta.y + s) * 2 + 1].xyz;
        mn = min(mn, bmn);
        mx = max(mx, bmx);
    }
    const float3 extents = (mx - mn) * 0.5f;
    const float minMargin = calculatePCMConvexMargin(extents, cs.scale.xyz, toleranceLength) * 0.5f;
    const float ratio = 0.1f;
    const float breakingThresholdRatio = 0.5f;

    // trimeshToPlane = planeTransform.transformInv(trimeshTransform)
    const PxgTransform28 planeT = makeTransform28(cs.planeQ, cs.planeP.xyz);
    const PxgTransform28 meshT = makeTransform28(cs.meshQ, cs.meshP.xyz);
    const PxgTransform28 trimeshToPlane = transformInvTransform(planeT, meshT);

    // 2. manifold gate; the previous relative transform is the manifold's own
    //    mRelativeTransform (gManRel), exactly as invalidateManifold reads it
    const float4 prevQ = gManRel[ci * 2 + 0];
    const float3 prevP = gManRel[ci * 2 + 1].xyz;
    const float thresholdP = minMargin * ratio;
    const bool invalidate = tpInvalidateManifold(trimeshToPlane.q(), trimeshToPlane.p(),
                                                 prevQ, prevP, thresholdP);

    uint4 manMeta = gManMeta[ci];
    bool lostContacts = false;
    if (!invalidate)
    {
        const float thr = minMargin * breakingThresholdRatio;
        const float sqThr = thr * thr;
        const uint base = ci * 24;
        for (uint s = 0; s < 24; ++s)
        {
            const uint mi = s / 6;
            if (mi >= manMeta.x) continue;
            const uint nb = mi == 0 ? manMeta.y : mi == 1 ? manMeta.z : mi == 2 ? manMeta.w : 0;
            if (s % 6 >= nb) continue;
            const float3 pointA = gManPointA[base + s].xyz;
            const float3 pointB = gManPointB[base + s].xyz;
            float4 nPen = gManNPen[base + s];
            const bool lost = tpRefreshContact(trimeshToPlane.q(), trimeshToPlane.p(),
                                               pointA, pointB, nPen, sqThr);
            gManNPen[base + s] = nPen;
            if (lost)
                lostContacts = true;
        }
        if (lostContacts)
            manMeta.x = 0;
    }
    const bool doFullContactGen = invalidate || lostContacts;

    const float3 worldNormal = quatBasisVector0(cs.planeQ);

    // shared contacts buffer: (32 + 1) * 6 slots
    float3 contacts[198];
    float seps[198];
    uint retainedCount = 0;

    if (doFullContactGen)
    {
        const uint numVerts = cs.verts.y;
        const uint numIterationsRequired = (numVerts + 1023) / 1024;
        for (uint iter = 0; iter < numIterationsRequired; ++iter)
        {
            // stage 1: 32 warps, each reducing its 32 lanes
            float3 warpPts[32][6];
            float warpSeps[32][6];
            uint counters[32];
            for (uint w = 0; w < 32; ++w)
            {
                float3 lanePt[32];
                float laneSep[32];
                uint mask = 0;
                for (uint l = 0; l < 32; ++l)
                {
                    lanePt[l] = float3(0, 0, 0);
                    laneSep[l] = FLT_MAX_LOCAL;
                    const uint i = iter * 1024 + w * 32 + l;
                    if (i >= numVerts) continue;
                    const float3 tp = gVerts[cs.verts.x + i].xyz;
                    const float3 p = vertex2Shape(tp, cs.scale.xyz, cs.scaleRot);
                    const float3 pInPlaneSpace = quatRotate(trimeshToPlane.q(), p) + trimeshToPlane.p();
                    if (cDistance >= pInPlaneSpace.x) // INCLUSIVE
                    {
                        laneSep[l] = pInPlaneSpace.x;
                        lanePt[l] = quatRotate(cs.meshQ, p) + cs.meshP.xyz;
                        mask |= 1u << l;
                    }
                }
                mask = contactReduce6(lanePt, laneSep, worldNormal, mask, clusterTolerance);
                counters[w] = countbits(mask);
                uint k = 0;
                for (uint l = 0; l < 32; ++l)
                {
                    if (((mask >> l) & 1) == 0) continue;
                    warpPts[w][k] = lanePt[l];
                    warpSeps[w][k] = laneSep[l];
                    ++k;
                }
            }
            // block scan: exclusive prefix of warp counters, contacts behind retainedCount
            uint offset = 0;
            for (uint w = 0; w < 32; ++w)
            {
                for (uint k = 0; k < counters[w]; ++k)
                {
                    contacts[retainedCount + offset + k] = warpPts[w][k];
                    seps[retainedCount + offset + k] = warpSeps[w][k];
                }
                offset += counters[w];
            }
            const uint totalContacts = offset + retainedCount;

            // stage 2: warp-0 readMask batching loop, in pxgSlangTriPlane
            retainedCount = tpBatchReduce(contacts, seps, totalContacts, worldNormal,
                                          clusterTolerance);
        }

        // 4. PCM store
        const uint base = ci * 24;
        if (retainedCount != 0)
        {
            // CUDA writes mNbManifolds and mNbContacts[0] only; [1..3] stay stale
            manMeta.x = 1;
            manMeta.y = retainedCount;
            gManRel[ci * 2 + 0] = trimeshToPlane.q();
            gManRel[ci * 2 + 1] = float4(trimeshToPlane.p(), 0);
            for (uint k = 0; k < retainedCount; ++k)
            {
                const float3 world = contacts[k];
                const float3 pointA = quatRotateInv(cs.meshQ, world - cs.meshP.xyz);
                float3 planePoint = quatRotateInv(cs.planeQ, world - cs.planeP.xyz);
                planePoint.x = 0.0f;
                gManPointA[base + k] = float4(pointA, asfloat(0xFFFFFFFFu)); // triIndex
                gManPointB[base + k] = float4(planePoint, 0);
                gManNPen[base + k] = float4(1.0f, 0.0f, 0.0f, seps[k]);
            }
        }
        else
            manMeta.x = 0;
    }
    else
    {
        // 5. cached: reload manifold 0 (post-refresh penetrations)
        if (manMeta.x != 0)
        {
            const uint base = ci * 24;
            // CUDA loads with one warp (threadIdx.x < mNbContacts[0]): at
            // most 32 slots even for hostile counts; retainedCount stays raw
            for (uint k = 0; k < manMeta.y && k < 32u; ++k)
            {
                const float3 pointA = gManPointA[base + k].xyz;
                contacts[k] = quatRotate(cs.meshQ, pointA) + cs.meshP.xyz;
                seps[k] = gManNPen[base + k].w;
            }
            retainedCount = manMeta.y;
        }
    }
    gManMeta[ci] = manMeta;

    // 6. contact-stream tail
    uint nbContacts = retainedCount;
    uint contactByteOffset = 0xFFFFFFFFu;
    uint forceAndIndiceByteOffset = 0xFFFFFFFFu;
    if (nbContacts != 0)
    {
        InterlockedAdd(gCounters[0], 16u * nbContacts, contactByteOffset);
        InterlockedAdd(gCounters[1], 4u * nbContacts, forceAndIndiceByteOffset);
        if ((contactByteOffset + 16u * nbContacts) > gParams.contactBytesLimit)
        {
            InterlockedOr(gCounters[2], OVERFLOW_CONTACT);
            contactByteOffset = 0xFFFFFFFFu;
        }
        if ((forceAndIndiceByteOffset + 4u * nbContacts) > gParams.forceBytesLimit)
        {
            InterlockedOr(gCounters[2], OVERFLOW_FORCE);
            forceAndIndiceByteOffset = 0xFFFFFFFFu;
            contactByteOffset = 0xFFFFFFFFu; // 5.12: a force overflow also drops the contacts
        }
    }
    uint4 rec;
    if (contactByteOffset != 0xFFFFFFFFu)
    {
        // CUDA writes with one warp (threadIndexInWarp < nbContacts): at
        // most 32 float4s; the reservation and the record keep the raw count
        for (uint k = 0; k < nbContacts && k < 32u; ++k)
            gContactStream[contactByteOffset / 16u + k] = float4(contacts[k], seps[k]);
        rec = uint4(contactByteOffset, forceAndIndiceByteOffset, nbContacts, doFullContactGen ? 1 : 0);
    }
    else
    {
        rec = uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0, doFullContactGen ? 1 : 0); // NULL pointers, nbContacts zeroed
        nbContacts = 0;
    }
    gOutRec[ci] = rec;
    gDebug[ci] = float4(worldNormal, float(retainedCount));
}
