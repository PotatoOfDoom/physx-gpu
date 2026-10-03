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
// ONE THREAD SERIALIZES THE BLOCK
// ---------------------------------------------------------------------------
// The CUDA spreads one contact manager across 32 warps: ballots, a block scan over
// the 32 warp counters and a warp-0 batching loop over the shared buffer.  The
// module's core is the SERIAL transcription of that -- ballots became lane arrays,
// the block scan a prefix sum, the shared buffer a local array -- and it is the
// version Phase 1 certified bit-exactly.  So this kernel runs it on thread 0 and
// lets the rest of the group return.
//
// [numthreads(32,1,1)] AND WHY THAT IS NOT THE P9.68 DEFECT.  That checker exists
// because a host which sizes its grid as `ceil(n / cudaBlock)` and a shader with a
// smaller [numthreads] reach a fixed FRACTION of the work forever.  The rule does
// not apply here: this launcher sizes its grid as `numBlocks = numTests`
// (PxgNarrowphaseCore.cpp:1261), one block per contact manager and not derived
// from the block size at all, so every pair still gets its own group whatever the
// group is worth.  Declaring 1024 would leave 1023 threads returning immediately
// on llvmpipe for no coverage gained.
//
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

[shader("compute")]
[numthreads(32, 1, 1)]
void trimeshPlaneNarrowphase(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    if (gtid.x != 0u)
        return;                       // thread 0 serializes the block (see the header)

    const uint workIndex = gid.x;

    PxgShapeGpu meshShape, planeShape;
    uint meshCacheRef, planeCacheRef;
    loadTrimeshPlanePair(workIndex, meshShape, meshCacheRef, planeShape, planeCacheRef);

    Ptr<float> contactDistance = Ptr<float>(gTP.contactDistance);
    const float cDistance = contactDistance[meshCacheRef] + contactDistance[planeCacheRef];

    Ptr<PxsCachedTransformGpu> xforms = Ptr<PxsCachedTransformGpu>(gTP.transformCache);
    const PxsCachedTransformGpu meshXf = xforms[meshCacheRef];
    const PxsCachedTransformGpu planeXf = xforms[planeCacheRef];
    const float4 meshQ = float4(meshXf.qx, meshXf.qy, meshXf.qz, meshXf.qw);
    const float3 meshP = float3(meshXf.px, meshXf.py, meshXf.pz);
    const float4 planeQ = float4(planeXf.qx, planeXf.qy, planeXf.qz, planeXf.qw);
    const float3 planeP = float3(planeXf.px, planeXf.py, planeXf.pz);

    // readTriangleMesh (cudaNpCommon.h): header uint4 | BV32DataPacked[header.w] |
    // float4 verts[header.x] | uint4 indices.  header.x IS numVerts, which is what
    // `counts.x` is read as at trimeshCollision.cu:118.
    const uint64_t meshPtr = meshShape.hullOrMeshPtr;
    const uint4 meshHeader = Ptr<uint4>(meshPtr)[0];
    const uint numVerts = meshHeader.x;
    const uint64_t treeAddr = meshPtr + 16UL;
    const uint64_t vertsAddr = treeAddr + uint64_t(meshHeader.w) * BV32_STRIDE;
    Ptr<float4> meshVerts = Ptr<float4>(vertsAddr);

    // The extents come from the ROOT node's child bounds, min/max over its
    // mNbNodes slots -- the CUDA gives one lane per slot and then reduces with
    // minIndex/maxIndex over the FULL mask, with the lanes past mNbNodes seeded at
    // +/-PX_MAX_F32 so they never win.  Serialized, that is exactly this loop.
    const uint nbRootSlots = Ptr<uint>(treeAddr + BV32_MNBNODES_OFFSET)[0];
    float3 mn = float3(FLT_MAX_LOCAL, FLT_MAX_LOCAL, FLT_MAX_LOCAL);
    float3 mx = float3(-FLT_MAX_LOCAL, -FLT_MAX_LOCAL, -FLT_MAX_LOCAL);
    for (uint s = 0u; s < min(nbRootSlots, 32u); ++s)
    {
        mn = min(mn, bvMin(treeAddr, 0u, s));
        mx = max(mx, bvMax(treeAddr, 0u, s));
    }
    const float3 extents = (mx - mn) * 0.5f;
    const float3 meshScale = float3(meshShape.scaleX, meshShape.scaleY, meshShape.scaleZ);
    const float4 meshScaleRot = float4(meshShape.scaleRotX, meshShape.scaleRotY,
                                       meshShape.scaleRotZ, meshShape.scaleRotW);
    const float minMargin =
        calculatePCMConvexMargin(extents, meshScale, gTP.toleranceLength) * 0.5f;
    const float ratio = 0.1f;
    const float breakingThresholdRatio = 0.5f;

    // trimeshToPlane = planeTransform.transformInv(trimeshTransform) (trimeshCollision.cu:147),
    // spelled as PxTransform::transformInv(PxTransform) is (PxTransform.h:162-169):
    // qinv = q.getConjugate(); (qinv.rotate(src.p - p), qinv * src.q).  P9.94d: NOT
    // q.rotateInv(v) -- same value, but an exact-zero component can take the other
    // sign, and relP is STORED (PCM_RELTRANSFORM_P below).
    const float4 planeQinv = float4(-planeQ.x, -planeQ.y, -planeQ.z, planeQ.w);
    const float4 relQ = npAbiQuatMul(planeQinv, meshQ);
    const float3 relP = npAbiQuatRotate(planeQinv, meshP - planeP);

    const uint64_t manifold = gTP.multiManifold + uint64_t(workIndex) * PCM_STRIDE;

    bool lostContacts = false;
    const bool invalidate = mmInvalidateManifold(manifold, relQ, relP, minMargin, ratio);
    if (!invalidate)
    {
        const float projectBreakingThreshold = minMargin * breakingThresholdRatio;
        // 24 SLOTS, not 24 lanes -- see the header.  The OR is the CUDA's
        // __any_sync over the warp.
        for (uint lane = 0u; lane < PCM_MAX_SLOTS; ++lane)
            if (mmRefreshSlot(manifold, relQ, relP, projectBreakingThreshold, lane))
                lostContacts = true;
        if (lostContacts)
            Ptr<uint>(manifold + PCM_NBMANIFOLDS)[0] = 0u;
    }
    const bool doFullContactGen = invalidate || lostContacts;

    const float3 worldNormal = quatBasisVector0(planeQ);

    float3 contacts[198];
    float seps[198];
    uint retainedCount = 0u;

    if (doFullContactGen)
    {
        const uint numIterationsRequired = (numVerts + 1023u) / 1024u;
        for (uint iter = 0u; iter < numIterationsRequired; ++iter)
        {
            // stage 1: 32 warps, each reducing its 32 lanes
            float3 warpPts[32][6];
            float warpSeps[32][6];
            uint counters[32];
            for (uint w = 0u; w < 32u; ++w)
            {
                float3 lanePt[32];
                float laneSep[32];
                uint mask = 0u;
                for (uint l = 0u; l < 32u; ++l)
                {
                    lanePt[l] = float3(0, 0, 0);
                    laneSep[l] = FLT_MAX_LOCAL;
                    const uint i = iter * 1024u + w * 32u + l;
                    if (i >= numVerts) continue;
                    const float3 tp = meshVerts[i].xyz;
                    const float3 p = vertex2Shape(tp, meshScale, meshScaleRot);
                    const float3 pInPlaneSpace = npAbiQuatRotate(relQ, p) + relP;
                    if (cDistance >= pInPlaneSpace.x)   // INCLUSIVE, as upstream
                    {
                        laneSep[l] = pInPlaneSpace.x;
                        lanePt[l] = npAbiQuatRotate(meshQ, p) + meshP;
                        mask |= 1u << l;
                    }
                }
                mask = contactReduce6(lanePt, laneSep, worldNormal, mask, gTP.clusterTolerance);
                counters[w] = countbits(mask);
                uint k = 0u;
                for (uint l = 0u; l < 32u; ++l)
                {
                    if (((mask >> l) & 1u) == 0u) continue;
                    warpPts[w][k] = lanePt[l];
                    warpSeps[w][k] = laneSep[l];
                    ++k;
                }
            }
            // block scan: exclusive prefix of the warp counters, BEHIND retainedCount
            uint offset = 0u;
            for (uint w = 0u; w < 32u; ++w)
            {
                for (uint k = 0u; k < counters[w]; ++k)
                {
                    contacts[retainedCount + offset + k] = warpPts[w][k];
                    seps[retainedCount + offset + k] = warpSeps[w][k];
                }
                offset += counters[w];
            }
            const uint totalContacts = offset + retainedCount;

            // stage 2: the warp-0 readMask batching loop, in pxgSlangTriPlane
            retainedCount = tpBatchReduce(contacts, seps, totalContacts, worldNormal,
                                          gTP.clusterTolerance);
        }

        // ---- the PCM store -------------------------------------------------
        // The CUDA writes mNbManifolds and mNbContacts[0] only; [1..3] stay stale.
        if (retainedCount != 0u)
        {
            Ptr<uint>(manifold + PCM_NBMANIFOLDS)[0] = 1u;
            Ptr<uint>(manifold + PCM_NBCONTACTS_OFFSET)[0] = retainedCount;
            Ptr<float4>(manifold + PCM_RELTRANSFORM_Q)[0] = relQ;
            Ptr<float4>(manifold + PCM_RELTRANSFORM_P)[0] = float4(relP, 0.0f);
            for (uint k = 0u; k < retainedCount; ++k)
            {
                const float3 world = contacts[k];
                const float3 pointA = npAbiQuatRotateInv(meshQ, world - meshP);
                float3 planePoint = npAbiQuatRotateInv(planeQ, world - planeP);
                planePoint.x = 0.0f;                     // FORCED onto the plane
                const uint64_t c = manifold + uint64_t(k) * PCM_CONTACT_STRIDE;
                storeF3(c + PCM_C_POINTA, pointA);
                storeF3(c + PCM_C_POINTB, planePoint);
                storeF3(c + PCM_C_NORMAL, float3(1.0f, 0.0f, 0.0f));  // PLANE SPACE
                Ptr<float>(c + PCM_C_PEN)[0] = seps[k];
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
                contacts[k] = npAbiQuatRotate(meshQ, pointA) + meshP;
                seps[k] = Ptr<float>(c + PCM_C_PEN)[0];
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
                           workIndex, nbContacts, forceByteOffset);

    if (contactByteOffset != NPT_INVALID)
    {
        Ptr<float4> baseContactStream =
            Ptr<float4>(gTP.contactStream + uint64_t(contactByteOffset));
        // the CUDA writes with one warp (threadIndexInWarp < nbContacts): at most
        // 32 float4s, while the reservation and the record keep the raw count
        for (uint k = 0u; k < nbContacts && k < 32u; ++k)
            baseContactStream[k] = float4(contacts[k], seps[k]);
    }
    else
    {
        nbContacts = 0u;
    }

    const uint patchIndex =
        nptRegisterContactPatch(gTP.patchAndContactCounters, gTP.cmOutputs,
                                gTP.touchChangeFlags, gTP.patchChangeFlags,
                                gTP.startContactPatches, gTP.patchBytesLimit,
                                workIndex, nbContacts);

    // `if (nbContacts == 0) cmOutputs[workIndex].contactPatches = 0` -- upstream
    // clears the patch pointer AFTER registering, which registerContactPatch does
    // not do for it (trimeshCollision.cu:422-423).
    if (nbContacts == 0u)
        Ptr<PxsContactManagerOutputGpu>(gTP.cmOutputs)[workIndex].contactPatches = 0uL;

    nptInsertIntoPatchStream(gTP.materials, gTP.patchStream,
                             planeShape.materialIndex, meshShape.materialIndex,
                             patchIndex, worldNormal, nbContacts);
}
