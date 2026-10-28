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

// triPlaneNpBda.slang -- P9.82.
//
//   trimeshPlaneNarrowphase (trimeshCollision.cu:63-430) on the engine launch ABI.
//
// One block per contact manager.  The mesh's vertices are tested against the
// plane, the survivors reduced to at most six, stored in the persistent manifold
// and emitted to the contact and patch streams.
//
// ---------------------------------------------------------------------------
// THIS KERNEL IS ASSEMBLY, NOT A NEW PORT, AND THAT IS DELIBERATE
// ---------------------------------------------------------------------------
// Phase 1 had already transcribed the whole algorithm into npTriPlaneNp.slang and
// certified it BIT-EXACTLY against a CPU reference.  Writing a second
// transcription here would have produced exactly the pair P9.15d cost a tranche to
// learn about: a fixture certifying one copy while the engine runs another.  So
// P9.82 first MOVED the shared parts into modules and this file composes them:
//
//   pxgSlangTriPlane     the decision core -- contactReduce6 with its clustering,
//                        the batching loop's faithful `ind += popc(readMask)`
//                        overshoot, calculatePCMConvexMargin, vertex2Shape
//   pxgSlangPcmManifold  the 1216-byte record, mmInvalidateManifold, mmRefreshSlot
//   pxgSlangNpPatchTail  the reservation, registerContactPatch, insertIntoPatchStream
//   pxgSlangBv32         the root node's child bounds
//
// What is genuinely NEW here is only the glue Phase 1 listed as not ported: the
// shape-pair load, the cooked-blob walk, the grid mapping and the stream tail's
// wiring.
//
// ---------------------------------------------------------------------------
// WHY NO SCENE COULD REACH IT
// ---------------------------------------------------------------------------
// gTable[ePLANE][eTRIANGLEMESH] is cndCasePlaneMesh
// (PxgNphaseImplementationContext.cpp:1052-1059), which returns eTrianglePlane
// whenever the mesh side is rigid -- with NO GPU-compatibility test, unlike the
// sphere/capsule/box/convex-vs-mesh cases.  The gate is upstream of that: PhysX 5.6
// will not simulate a non-kinematic PxTriangleMeshGeometry without an SDF
// (NpRigidBodyTemplate.h:359-374), and no mesh in this tree had ever been cooked
// with one.  PXG_SCENE=sdfmesh (P9.82a) is the first, and its first run measured
// the body falling to -73.8070 against a rest height of 0.500 with this kernel
// launched 183 times and skipped 183 times.
//
// ---------------------------------------------------------------------------
// THE BLOCK SHAPE: 1024 THREADS, AS THE CUDA (and why it was one thread until now)
// ---------------------------------------------------------------------------
// The CUDA spreads one contact manager across 32 warps: every thread tests one
// vertex, each warp reduces its 32 lanes, a block scan over the 32 warp counters
// packs the survivors behind the carried retainedCount, and warp 0 re-reduces the
// shared buffer in a batching loop.  Until the 1024-thread rework this kernel ran
// the module's SERIAL transcription of all of that on thread 0 of a 32-thread
// group: 32 lane reductions and every vertex transform one after the other, with
// the whole shared buffer (198 points) and 32 x 6 warp survivors in private arrays.
// It ran only when the manifold was invalidated (the cached path is a reload), but
// then it was the whole block's work on one thread.
//
// Now the group is the CUDA's block:
//   stage 1a  thread t tests vertex iter*1024 + t (the CUDA's `i`), exactly the
//             per-lane plane test, and parks (point, separation) in groupshared;
//   stage 1b  lane 0 of each warp runs contactReduce6 over ITS warp's 32 lanes --
//             the same serial body on the same lane arrays, 32 reductions in
//             parallel instead of in sequence;
//   stage 1c  every surviving lane writes itself at retainedCount + the exclusive
//             prefix of the warp counters + its rank in the warp mask -- the
//             CUDA's `offset + contactWarpOffset + retainedCount`, warp-major,
//             lane order, i.e. the slots the serial pack produced;
//   stage 2   thread 0 runs the warp-0 batching loop (tpBatchReduceStore, the
//             module's one body, here over the groupshared buffer).
// Every value is produced by the same expression on the same operands as before,
// and every buffer slot is written in the same order, so the result is bit-for-bit
// the serial kernel's (npTriPlaneNp_engineAbiFleet pins it against refTp, which the
// serial kernel also passes).  The cached path and the stream tail stay on thread
// 0: they are a reload of at most 32 contacts and a handful of atomics.
//
// THE NUMTHREADS NOTE THIS REPLACES said 1024 would leave 1023 threads idle.  That
// was true of the serial body and is why it was 32 then; it is not a coverage
// question either way -- the launcher sizes its grid as `numBlocks = numTests`
// (PxgNarrowphaseCore.cpp), one block per contact manager, not derived from the
// block size.
//
// Barriers: the full-gen branch and its iteration count are group-uniform (a
// groupshared flag written by thread 0 behind a barrier, and numVerts read from
// the shared mesh blob), so every barrier below is reached by all 1024 threads.

// ---------------------------------------------------------------------------
// THE MANIFOLD GATE RUNS OVER 24 SLOTS, NOT 24 LANES
// ---------------------------------------------------------------------------
// refreshManifolds is written as `PxU32 i = threadIdxInWarp;` with no loop -- 24
// slots and at least 24 lanes, so the warp covers them all in one pass and ORs the
// per-lane `lostContact`.  Serialized, that is a loop over lanes 0..23 with the
// same OR, which is what mmRefreshSlot's `lane` parameter is for.  Lanes 24..31
// are past the last slot and mmRefreshSlot returns false for them; the loop stops
// at 24 rather than 32 because those lanes provably contribute nothing.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the plane-space projection, the
// reduction's anchor picks and the clustering distances are mul-feeding-add chains
// whose SIGN decides which contacts survive.

module triPlaneNpBda;

import pxgSlangCommon;
import pxgSlangNpAbi;
import pxgSlangBv32;
import pxgSlangPcmManifold;
import pxgSlangNpPatchTail;
import pxgSlangTriPlane;
import pxgSlangCvxTriSetup;   // CT_GEOM_TRIANGLEMESH -- one spelling of the enum

// *** PxGeometryType::eTRIANGLEMESH IS 8, NOT 4. ***  P9.82's first draft wrote 4,
// which is eCONVEXCORE, so `shapes[shapeRef0].type != GEOM_TRIANGLEMESH` was true for
// the trimesh, the pair came back SWAPPED, and the kernel dereferenced the PLANE's
// hullOrMeshPtr as a cooked-mesh blob.  The scene segfaulted on the first step.  This
// is the P9.57 defect class exactly: the enum is
//   eSPHERE 0, ePLANE 1, eCAPSULE 2, eBOX 3, eCONVEXCORE 4, eCONVEXMESH 5,
//   ePARTICLESYSTEM 6, eTETRAHEDRONMESH 7, eTRIANGLEMESH 8, eHEIGHTFIELD 9
// and every one of them is a plausible-looking small integer.
static const uint GEOM_TRIANGLEMESH = CT_GEOM_TRIANGLEMESH;   // = 8, from pxgSlangCvxTriSetup

// ---------------------------------------------------------------------------
// Push-constant blob -- the CUDA parameter list EXACTLY
// (PxgNarrowphaseCore.cpp:1236-1256): PxReal @0, fifteen pointers @8..@120, then
// four PxU32/PxReal scalars -> 144 bytes.
// ABI row { 4,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,4,4 }.
// ---------------------------------------------------------------------------
struct TriPlaneParams
{
    float toleranceLength;              // @0
    uint _pad4;                         // @4
    uint64_t cmInputs;                  // @8
    uint64_t cmOutputs;                 // @16
    uint64_t shapes;                    // @24
    uint64_t transformCache;            // @32
    uint64_t contactDistance;           // @40
    uint64_t materials;                 // @48
    uint64_t multiManifold;             // @56
    uint64_t contactStream;             // @64
    uint64_t patchStream;               // @72
    uint64_t patchAndContactCounters;   // @80
    uint64_t touchChangeFlags;          // @88
    uint64_t patchChangeFlags;          // @96
    uint64_t startContactPatches;       // @104
    uint64_t startContactPoints;        // @112
    uint64_t startContactForces;        // @120
    uint patchBytesLimit;               // @128
    uint contactBytesLimit;             // @132
    uint forceBytesLimit;               // @136
    float clusterTolerance;             // @140 -> 144  (the CUDA's clusterBias)
}

[[vk::push_constant]] ConstantBuffer<TriPlaneParams> gTP;

// LoadShapePairWarp<eTRIANGLEMESH, ePLANE>: the trimesh is whichever half of the
// pair IS the triangle mesh, the plane is the other.  Routed by geometry TYPE, not
// by position -- cndCasePlaneMesh sorts (ePLANE, eTRIANGLEMESH) so the plane is
// slot 0, but the kernel does not rely on that and neither does this.
void loadTrimeshPlanePair(uint cmIdx, out PxgShapeGpu meshShape, out uint meshCacheRef,
                          out PxgShapeGpu planeShape, out uint planeCacheRef)
{
    const PxgContactManagerInputGpu cin = Ptr<PxgContactManagerInputGpu>(gTP.cmInputs)[cmIdx];
    Ptr<PxgShapeGpu> shapes = Ptr<PxgShapeGpu>(gTP.shapes);
    uint sMesh = cin.shapeRef0, cMesh = cin.transformCacheRef0;
    uint sPlane = cin.shapeRef1, cPlane = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != GEOM_TRIANGLEMESH)
    {
        sMesh = cin.shapeRef1; cMesh = cin.transformCacheRef1;
        sPlane = cin.shapeRef0; cPlane = cin.transformCacheRef0;
    }
    meshShape = shapes[sMesh]; meshCacheRef = cMesh;
    planeShape = shapes[sPlane]; planeCacheRef = cPlane;
}

// ---------------------------------------------------------------------------
// The CUDA's shared buffers (trimeshCollision.cu: contacts / separations /
// counters / retainedCount / doFullContactGen), plus the per-lane staging the
// emulated __ballot_sync needs.  ~20 KB, under the groupshared watermark.
// ---------------------------------------------------------------------------
static const uint TP_THREADS = 1024u;
static const uint TP_BUFFER = 198u;   // (WARP_SIZE + 1) * PLANE_TRI_MAX_CONTACTS

groupshared float4 sTpPtSep[TP_THREADS];   // stage 1: each lane's (worldPoint, separation)
groupshared uint sTpHasMask[32];           // stage 1: __ballot_sync(hasContact) per warp
groupshared uint sTpWarpMask[32];          // stage 1: each warp's reduced mask
groupshared float4 sTpContacts[TP_BUFFER]; // contacts / separations, packed (point, sep)
groupshared uint sTpRetained;              // retainedCount
groupshared uint sTpFullGen;               // doFullContactGen

// The batching loop's view of the groupshared buffer.  The clamp bounds only the
// address llvmpipe forms for lanes the `readIndex < totalContacts` guard excludes
// (P9.22: guarded loads are issued unpredicated); every live index is < TP_BUFFER.
struct TpSharedStore : ITpContactStore
{
    float3 tpPoint(uint i) { return sTpContacts[min(i, TP_BUFFER - 1u)].xyz; }
    float tpSep(uint i) { return sTpContacts[min(i, TP_BUFFER - 1u)].w; }
    [mutating] void tpStore(uint i, float3 p, float s) { sTpContacts[i] = float4(p, s); }
}

// Everything the vertex test needs, published by thread 0 behind the gate's barrier:
// the rest of the group reads COPIES of thread 0's values, so they are the same bits,
// and on the cached path (the common one) the other 1023 threads do nothing but meet
// that barrier and leave.
groupshared float4 sTpQ[3];     // relQ, meshQ, meshScaleRot
groupshared float4 sTpV[4];     // (relP, cDistance), (meshP, -), (meshScale, -), (worldNormal, -)
groupshared uint4 sTpU;         // numVerts, the vertex array's address (lo, hi)

// The pair, its poses and the derived relative transform: thread 0's prologue.
struct TpPair
{
    PxgShapeGpu meshShape, planeShape;
    float cDistance;
    float4 meshQ, planeQ, relQ;
    float3 meshP, planeP, relP;
    float3 meshScale;
    float4 meshScaleRot;
    uint numVerts;
    uint64_t treeAddr, vertsAddr;
}

TpPair tpLoadPair(uint pairIdx)
{
    TpPair pr;
    uint meshCacheRef, planeCacheRef;
    loadTrimeshPlanePair(pairIdx, pr.meshShape, meshCacheRef, pr.planeShape, planeCacheRef);

    Ptr<float> contactDistance = Ptr<float>(gTP.contactDistance);
    pr.cDistance = contactDistance[meshCacheRef] + contactDistance[planeCacheRef];

    Ptr<PxsCachedTransformGpu> xforms = Ptr<PxsCachedTransformGpu>(gTP.transformCache);
    const PxsCachedTransformGpu meshXf = xforms[meshCacheRef];
    const PxsCachedTransformGpu planeXf = xforms[planeCacheRef];
    pr.meshQ = float4(meshXf.qx, meshXf.qy, meshXf.qz, meshXf.qw);
    pr.meshP = float3(meshXf.px, meshXf.py, meshXf.pz);
    pr.planeQ = float4(planeXf.qx, planeXf.qy, planeXf.qz, planeXf.qw);
    pr.planeP = float3(planeXf.px, planeXf.py, planeXf.pz);

    // readTriangleMesh (cudaNpCommon.h): header uint4 | BV32DataPacked[header.w] |
    // float4 verts[header.x] | uint4 indices.  header.x IS numVerts, which is what
    // `counts.x` is read as at trimeshCollision.cu:118.
    const uint64_t meshPtr = pr.meshShape.hullOrMeshPtr;
    const uint4 meshHeader = Ptr<uint4>(meshPtr)[0];
    pr.numVerts = meshHeader.x;
    pr.treeAddr = meshPtr + 16UL;
    pr.vertsAddr = pr.treeAddr + uint64_t(meshHeader.w) * BV32_STRIDE;

    pr.meshScale = float3(pr.meshShape.scaleX, pr.meshShape.scaleY, pr.meshShape.scaleZ);
    pr.meshScaleRot = float4(pr.meshShape.scaleRotX, pr.meshShape.scaleRotY,
                             pr.meshShape.scaleRotZ, pr.meshShape.scaleRotW);

    // trimeshToPlane = planeTransform.transformInv(trimeshTransform) (trimeshCollision.cu:147),
    // spelled as PxTransform::transformInv(PxTransform) is (PxTransform.h:162-169):
    // qinv = q.getConjugate(); (qinv.rotate(src.p - p), qinv * src.q).  P9.94d: NOT
    // q.rotateInv(v) -- same value, but an exact-zero component can take the other
    // sign, and relP is STORED (PCM_RELTRANSFORM_P below).
    const float4 planeQinv = float4(-pr.planeQ.x, -pr.planeQ.y, -pr.planeQ.z, pr.planeQ.w);
    pr.relQ = npAbiQuatMul(planeQinv, pr.meshQ);
    pr.relP = npAbiQuatRotate(planeQinv, pr.meshP - pr.planeP);
    return pr;
}

[shader("compute")]
[numthreads(1024, 1, 1)]   // TP_THREADS, the CUDA block (see the header)
void trimeshPlaneNarrowphase(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint tid = gtid.x;
    const uint warpIdx = tid >> 5u;
    const uint lane = tid & 31u;
    // The CUDA's workIndex = blockIdx.x: one block per contact manager, uniform
    // across the group.  Named pairIdx so the barrier checker does not read it as a
    // warp index, which is what `workIndex` means in the midphase kernels.
    const uint pairIdx = gid.x;

    const uint64_t manifold = gTP.multiManifold + uint64_t(pairIdx) * PCM_STRIDE;

    // ---- the prologue and the manifold gate: warp 0 in the CUDA, thread 0 here ----
    if (tid == 0u)
    {
        const TpPair pr = tpLoadPair(pairIdx);

        // The extents come from the ROOT node's child bounds, min/max over its
        // mNbNodes slots -- the CUDA gives one lane per slot and then reduces with
        // minIndex/maxIndex over the FULL mask, with the lanes past mNbNodes seeded at
        // +/-PX_MAX_F32 so they never win.  Serialized, that is exactly this loop.
        const uint nbRootSlots = Ptr<uint>(pr.treeAddr + BV32_MNBNODES_OFFSET)[0];
        float3 mn = float3(FLT_MAX_LOCAL, FLT_MAX_LOCAL, FLT_MAX_LOCAL);
        float3 mx = float3(-FLT_MAX_LOCAL, -FLT_MAX_LOCAL, -FLT_MAX_LOCAL);
        for (uint s = 0u; s < min(nbRootSlots, 32u); ++s)
        {
            mn = min(mn, bvMin(pr.treeAddr, 0u, s));
            mx = max(mx, bvMax(pr.treeAddr, 0u, s));
        }
        const float3 extents = (mx - mn) * 0.5f;
        const float minMargin =
            calculatePCMConvexMargin(extents, pr.meshScale, gTP.toleranceLength) * 0.5f;
        const float ratio = 0.1f;
        const float breakingThresholdRatio = 0.5f;

        bool lostContacts = false;
        const bool invalidate = mmInvalidateManifold(manifold, pr.relQ, pr.relP, minMargin, ratio);
        if (!invalidate)
        {
            const float projectBreakingThreshold = minMargin * breakingThresholdRatio;
            // 24 SLOTS, not 24 lanes -- see the header.  The OR is the CUDA's
            // __any_sync over the warp.
            for (uint slot = 0u; slot < PCM_MAX_SLOTS; ++slot)
                if (mmRefreshSlot(manifold, pr.relQ, pr.relP, projectBreakingThreshold, slot))
                    lostContacts = true;
            if (lostContacts)
                Ptr<uint>(manifold + PCM_NBMANIFOLDS)[0] = 0u;
        }
        sTpFullGen = (invalidate || lostContacts) ? 1u : 0u;
        sTpRetained = 0u;

        sTpQ[0] = pr.relQ;
        sTpQ[1] = pr.meshQ;
        sTpQ[2] = pr.meshScaleRot;
        sTpV[0] = float4(pr.relP, pr.cDistance);
        sTpV[1] = float4(pr.meshP, 0.0f);
        sTpV[2] = float4(pr.meshScale, 0.0f);
        sTpV[3] = float4(quatBasisVector0(pr.planeQ), 0.0f);
        sTpU = uint4(pr.numVerts, uint(pr.vertsAddr), uint(pr.vertsAddr >> 32), 0u);
    }
    if (tid < 32u)
        sTpHasMask[tid] = 0u;
    GroupMemoryBarrierWithGroupSync();   // the verdict, the inputs and the cleared ballots are visible

    const bool doFullContactGen = sTpFullGen != 0u;   // group-uniform

    if (doFullContactGen)
    {
        const float4 relQ = sTpQ[0];
        const float4 meshQ = sTpQ[1];
        const float4 meshScaleRot = sTpQ[2];
        const float3 relP = sTpV[0].xyz;
        const float cDistance = sTpV[0].w;
        const float3 meshP = sTpV[1].xyz;
        const uint4 meshInfo = sTpU;
        const uint numVerts = meshInfo.x;
        const float3 meshScale = sTpV[2].xyz;
        const float3 worldNormal = sTpV[3].xyz;
        Ptr<float4> meshVerts = Ptr<float4>(uint64_t(meshInfo.y) | (uint64_t(meshInfo.z) << 32));

        const uint numIterationsRequired = (numVerts + TP_THREADS - 1u) / TP_THREADS;
        for (uint iter = 0u; iter < numIterationsRequired; ++iter)
        {
    // ---- stage 1a: one vertex per thread (trimeshCollision.cu:166-196) ----
            const uint i = iter * TP_THREADS + tid;
            float3 worldPoint = float3(0, 0, 0);
            float separation = FLT_MAX_LOCAL;
            bool hasContact = false;
            if (i < numVerts)
            {
                // The clamp bounds the address the masked-off SIMD lanes form (P9.22);
                // a live lane's index is unchanged.
                const float3 tp = meshVerts[min(i, numVerts - 1u)].xyz;
                const float3 p = vertex2Shape(tp, meshScale, meshScaleRot);
                const float3 pInPlaneSpace = npAbiQuatRotate(relQ, p) + relP;
                if (cDistance >= pInPlaneSpace.x)   // INCLUSIVE, as upstream
                {
                    separation = pInPlaneSpace.x;
                    worldPoint = npAbiQuatRotate(meshQ, p) + meshP;
                    hasContact = true;
                }
            }
            sTpPtSep[tid] = float4(worldPoint, separation);
            if (hasContact)
                InterlockedOr(sTpHasMask[warpIdx], 1u << lane);
            GroupMemoryBarrierWithGroupSync();   // every lane's point and ballot bit are in

            // ---- stage 1b: contactReduce<true,true,6,false> per warp, on its lane 0 ----
            // The lane arrays are exactly the serial kernel's: a lane without a contact
            // holds (0,0,0) / PX_MAX_F32.
            if (lane == 0u)
            {
                float3 lanePt[32];
                float laneSep[32];
                for (uint l = 0u; l < 32u; ++l)
                {
                    const float4 v = sTpPtSep[tid + l];
                    lanePt[l] = v.xyz;
                    laneSep[l] = v.w;
                }
                const uint ballot = sTpHasMask[warpIdx];
                sTpHasMask[warpIdx] = 0u;        // read once; cleared for the next iteration
                sTpWarpMask[warpIdx] =
                    contactReduce6(lanePt, laneSep, worldNormal, ballot, gTP.clusterTolerance);
            }
            GroupMemoryBarrierWithGroupSync();   // the 32 warp masks are in

            // ---- stage 1c: pack the survivors behind retainedCount (block scan) ----
            const uint retainedBefore = sTpRetained;
            const uint warpMask = sTpWarpMask[warpIdx];
            if (((warpMask >> lane) & 1u) != 0u)
            {
                uint contactWarpOffset = 0u;      // exclusive runsum of the warp counters
                for (uint w = 0u; w < warpIdx; ++w)
                    contactWarpOffset += countbits(sTpWarpMask[w]);
                const uint offset = countbits(warpMask & ((1u << lane) - 1u));
                sTpContacts[offset + contactWarpOffset + retainedBefore] = sTpPtSep[tid];
            }
            GroupMemoryBarrierWithGroupSync();   // the shared buffer is complete

            // ---- stage 2: the warp-0 batching loop, in pxgSlangTriPlane ----
            if (tid == 0u)
            {
                uint totalContacts = retainedBefore;
                for (uint w = 0u; w < 32u; ++w)
                    totalContacts += countbits(sTpWarpMask[w]);
                TpSharedStore store;
                sTpRetained = tpBatchReduceStore(store, totalContacts, worldNormal,
                                                 gTP.clusterTolerance);
            }
            GroupMemoryBarrierWithGroupSync();   // survivors compacted, retainedCount set
        }
    }

    // Everything below is warp 0 / thread 0 in the CUDA and has no barrier.
    if (tid != 0u)
        return;

    const TpPair pr = tpLoadPair(pairIdx);   // thread 0's prologue values, recomputed
    const float4 meshQ = pr.meshQ, planeQ = pr.planeQ;
    const float3 meshP = pr.meshP, planeP = pr.planeP;

    uint retainedCount = sTpRetained;

    if (doFullContactGen)
    {
        // ---- the PCM store -------------------------------------------------
        // The CUDA writes mNbManifolds and mNbContacts[0] only; [1..3] stay stale.
        if (retainedCount != 0u)
        {
            Ptr<uint>(manifold + PCM_NBMANIFOLDS)[0] = 1u;
            Ptr<uint>(manifold + PCM_NBCONTACTS_OFFSET)[0] = retainedCount;
            Ptr<float4>(manifold + PCM_RELTRANSFORM_Q)[0] = pr.relQ;
            Ptr<float4>(manifold + PCM_RELTRANSFORM_P)[0] = float4(pr.relP, 0.0f);
            for (uint k = 0u; k < retainedCount; ++k)
            {
                const float4 c4 = sTpContacts[k];
                const float3 world = c4.xyz;
                const float3 pointA = npAbiQuatRotateInv(meshQ, world - meshP);
                float3 planePoint = npAbiQuatRotateInv(planeQ, world - planeP);
                planePoint.x = 0.0f;                     // FORCED onto the plane
                const uint64_t c = manifold + uint64_t(k) * PCM_CONTACT_STRIDE;
                storeF3(c + PCM_C_POINTA, pointA);
                storeF3(c + PCM_C_POINTB, planePoint);
                storeF3(c + PCM_C_NORMAL, float3(1.0f, 0.0f, 0.0f));  // PLANE SPACE
                Ptr<float>(c + PCM_C_PEN)[0] = c4.w;
                Ptr<uint>(c + 40UL)[0] = 0xFFFFFFFFu;    // triIndex
            }
        }
        else
            Ptr<uint>(manifold + PCM_NBMANIFOLDS)[0] = 0u;
    }
    else
    {
        // ---- the CACHED path: reload manifold 0, POST-refresh penetrations ----
        // The CUDA loads with one warp guarded by `threadIdx.x < mNbContacts[0]`,
        // i.e. at most 32 slots even for a hostile count, while retainedCount keeps
        // the RAW value -- so a count above 32 reports more contacts than were
        // loaded.  Transcribed, as npTriPlaneNp transcribes it.
        const uint nbManifolds = Ptr<uint>(manifold + PCM_NBMANIFOLDS)[0];
        if (nbManifolds != 0u)
        {
            const uint nb = Ptr<uint>(manifold + PCM_NBCONTACTS_OFFSET)[0];
            for (uint k = 0u; k < nb && k < 32u; ++k)
            {
                const uint64_t c = manifold + uint64_t(k) * PCM_CONTACT_STRIDE;
                const float3 pointA = loadF3(c + PCM_C_POINTA);
                sTpContacts[k] = float4(npAbiQuatRotate(meshQ, pointA) + meshP,
                                        Ptr<float>(c + PCM_C_PEN)[0]);
            }
            retainedCount = nb;
        }
    }

    // ---- the contact-stream and patch tail, in pxgSlangNpPatchTail -----------
    uint nbContacts = retainedCount;
    uint forceByteOffset = NPT_INVALID;
    const uint contactByteOffset =
        nptReserveContacts(gTP.patchAndContactCounters, gTP.cmOutputs,
                           gTP.startContactPoints, gTP.startContactForces,
                           gTP.contactBytesLimit, gTP.forceBytesLimit,
                           pairIdx, nbContacts, forceByteOffset);

    if (contactByteOffset != NPT_INVALID)
    {
        Ptr<float4> baseContactStream =
            Ptr<float4>(gTP.contactStream + uint64_t(contactByteOffset));
        // the CUDA writes with one warp (threadIndexInWarp < nbContacts): at most
        // 32 float4s, while the reservation and the record keep the raw count
        for (uint k = 0u; k < nbContacts && k < 32u; ++k)
            baseContactStream[k] = sTpContacts[k];
    }
    else
    {
        nbContacts = 0u;
    }

    const uint patchIndex =
        nptRegisterContactPatch(gTP.patchAndContactCounters, gTP.cmOutputs,
                                gTP.touchChangeFlags, gTP.patchChangeFlags,
                                gTP.startContactPatches, gTP.patchBytesLimit,
                                pairIdx, nbContacts);

    // `if (nbContacts == 0) cmOutputs[workIndex].contactPatches = 0` -- upstream
    // clears the patch pointer AFTER registering, which registerContactPatch does
    // not do for it (trimeshCollision.cu:422-423).
    if (nbContacts == 0u)
        Ptr<PxsContactManagerOutputGpu>(gTP.cmOutputs)[pairIdx].contactPatches = 0uL;

    nptInsertIntoPatchStream(gTP.materials, gTP.patchStream,
                             pr.planeShape.materialIndex, pr.meshShape.materialIndex,
                             patchIndex, quatBasisVector0(planeQ), nbContacts);
}
