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

// triTriCollisionBda.slang -- P9.83b-8: triangleTriangleCollision
// (convexMeshMidphase.cu:1288-1377, with doTriangleTriangleCollision :972-1231).
//
// THE SDF MESH-MESH CONTACT GENERATOR, ONE 1024-THREAD WORKGROUP PER PAIR.  Every
// numeric and every block collective in it is an already-certified piece; this
// file is the glue the CUDA spreads over its wrapper and its driver:
//
//   pxgSlangTriTriGeom   the sub-triangle vertices, the two radii, the inverses
//   pxgSlangTriTriBody   the per-thread contact body of both passes
//   pxgSlangSdfCollision the SDF descent (doTriangleSDFCollision) and sdfDistance
//   pxgSlangTriTriFind   the subdivision-aware gather and the refinement buffer
//   pxgSlangTriTriBlock  tryToAssign / createNewPatches as block logic
//   pxgSlangTriTriTail   writePatchesAndContacts and the block-wide reservation
//
// What is new here, all of it small, all of it named below: the mesh-geometry
// reader, the overlap-byte read, doFirst and the one-body argument exchange,
// aToB and its materialised inverse, the fresh contact every iteration, the
// per-pass cullScale, and the index clamp at the body's reads.
//
// ---------------------------------------------------------------------------
// THE DRIVER'S TRAPS (K1-K9)
// ---------------------------------------------------------------------------
// K1  ONE BODY, ARGUMENTS EXCHANGED.  The CUDA has TWO call sites of
//     doTriangleTriangleCollision under `if (doFirst) / else` (:1363-1376) --
//     exactly one runs per block, the two differing only in operand order.  This
//     file selects the operands once and has one call site, so the 1024-thread
//     body is instantiated a single time.  isFlipped is doFirst.  The body sees LOCAL mesh0/mesh1: in
//     the flipped arm its "mesh0" is the engine's shape1.
// K2  doFirst IS A GUARD OVER THREE ALTERNATIVES (:1358-1360):
//         mesh0 has an SDF  AND  ( mesh1 has none
//                                  OR mesh1 prefers projection (sdfDims.w) and mesh0 not
//                                  OR the flags agree and mesh1 has FEWER vertices )
//     -- the SDF guard covers all three.  sdfDims.w is the projection FLAG, set
//     even for a mesh without an SDF (pxgSlangSdfTexture B3/B4); numVerts is
//     header .x, not the triangle count .y.
// K3  cullScale IS RESET EVERY PASS (:1036).  Pass 1 is contactDistance / PxMin
//     (shape0.scale), not pass 0's value divided again.  PxMin is selectMin,
//     `a < b ? a : b`, spelled, not the builtin (which is FMin, undefined on NaN).
// K4  PASS 1 RUNS ONLY WHEN THE LOCAL mesh0 HAS AN SDF (:1048-1055), and the body
//     does NOT return when a mesh has none: the local mesh0 is legitimately
//     SDF-less in the one-SDF (floor) arm.  The local mesh1 always has one --
//     the pair only reaches this bucket when at least one mesh does, and doFirst
//     puts it on the mesh1 side.
// K5  TWO INVERSES, KEPT APART (pxgSlangTriTriGeom).  The pass-1 GATHER is handed
//     the materialised aToB.getInverse() (:1078); the pass-1 BODY is handed aToB
//     itself and calls transformInv (:1165-1167).  Not the same arithmetic.
// K6  A FRESH CONTACT EVERY ITERATION (:1066): pos 0, separation PX_MAX_F32,
//     normal 0, candidate false.  `dir` starts at 0 before each descent: the CUDA
//     leaves it uninitialised and the descent's `inout` keeps it (Body T1).
// K7  THE REFINEMENT CALL IS OUTSIDE `if (ind < nb)` (:1198): it holds barriers.
// K8  THE LOOP TESTS THE REGISTER COPY (:1064 vs :1209), which the CUDA's shared
//     counter equals behind the :1204 barrier.  Waived in
//     tools/barrier_uniformity_waivers.txt, with the argument.
// K9  THE INDEX IS CLAMPED AT THE BODY'S READS (G2; F7 in pxgSlangTriTriFind).
//     An idle thread has ind = nbTriangles, and the guard is a mask on this
//     platform, so the vertex reads take `ind < nb ? ind : 0`.
//
// ---------------------------------------------------------------------------
// STORAGE: 0 bytes of staging.  The CUDA stages the shapes, transforms and meshes
// in shared memory (:1329-1354) and the SDF texture too (:1013-1014); here every
// thread holds its own copy, as triTriOverlapBda and the cloth SDF kernel do.
// The Workgroup storage is exactly what the modules reach -- measured in
// tools/groupshared_known.txt.
// ---------------------------------------------------------------------------

#define PXG_WARP_BLOCK 1024
#include "pxgSlangWarpInc.slang"

import pxgSlangCommon;
import pxgSlangNpAbi;
import pxgSlangSdfTexture;
import pxgSlangSdfCollision;
import pxgSlangContactReduceSerial;
import pxgSlangTriTriPatch;
import pxgSlangTriTriGeom;
import pxgSlangTriTriBody;

#include "pxgSlangSdfFind.slang"
#include "pxgSlangTriTriBlock.slang"
#include "pxgSlangTriTriFind.slang"
#include "pxgSlangTriTriTail.slang"

// ---------------------------------------------------------------------------
// The push blob -- the CUDA parameter list EXACTLY (convexMeshMidphase.cu:1290-
// 1311; host pack PxgNarrowphaseCore.cpp:1424-1446), 144 bytes.  The engine row
// { 4, 8 x 15, 4, 4, 4, 4 } is registered with the manifest entry (T6); until then
// the kernel is unmapped and check_launch_abi_pack.py cannot compare this struct
// with the pack -- npTriTriKernel's static_asserts on PushH are the only check.
// ---------------------------------------------------------------------------
struct TriTriCollisionParams
{
    uint numWorkItems;              // @0
    uint _pad4;                     // @4
    uint64_t cmInputs;              // @8    PxgContactManagerInput*
    uint64_t transformCache;        // @16   PxsCachedTransform*
    uint64_t contactDistances;      // @24   PxReal*
    uint64_t shapes;                // @32   PxgShape*
    uint64_t cmOutputs;             // @40   PxsContactManagerOutput*
    uint64_t overlaps;              // @48   PxU8*, one byte per pair
    uint64_t materials;             // @56   PxsMaterialData*
    uint64_t contactStream;         // @64
    uint64_t patchStream;           // @72
    uint64_t counters;              // @80   PxgPatchAndContactCounters*
    uint64_t touchChangeFlags;      // @88   PxU32*
    uint64_t patchChangeFlags;      // @96   PxU32*
    uint64_t startContactPatches;   // @104
    uint64_t startContactPoints;    // @112
    uint64_t startContactForces;    // @120
    uint patchBytesLimit;           // @128
    uint contactBytesLimit;         // @132
    uint forceBytesLimit;           // @136
    float clusterBias;              // @140
}

[[vk::push_constant]] ConstantBuffer<TriTriCollisionParams> gTc;

static const uint TTK_BV32_STRIDE = 1168u;                 // sizeof(Gu::BV32DataPacked)
static const uint TTK_MAX_REFINEMENT_LEVEL = 8u;           // :1059
static const float TTK_REFINEMENT_RATIO_SQUARED = 256.0f;  // :1060, 16 * 16 (exact)

// readTriangleMesh (triangleMesh.cuh:120-195): the geometry half here, the SDF
// half in pxgSlangSdfTexture (sdfReadMesh).  Header .x verts, .y tris, .w
// nbPackedNodes -- the count of BV32DataPacked records, which is what the tree
// occupies (triangleMesh.cuh:130, :133).  The CUDA names .z numBv32TreeNodes, but
// the only host writer stores 0 there (PxgGeometryManager.cpp:537), so .z is not
// the stride.  The float4 vertices follow the tree, the uint4 triangles follow the
// vertices.
struct TtkMesh
{
    uint numVerts;
    uint numTris;
    uint64_t verts;
    uint64_t indices;
    PxgSdfMesh sdf;
}

TtkMesh ttkReadMesh(uint64_t meshPtr)
{
    TtkMesh m;
    const uint4 hdr = Ptr<uint4>(meshPtr)[0];
    m.numVerts = hdr.x;
    m.numTris = hdr.y;
    m.verts = meshPtr + 16UL + uint64_t(hdr.w) * uint64_t(TTK_BV32_STRIDE);
    m.indices = m.verts + uint64_t(hdr.x) * 16UL;
    sdfReadMesh(meshPtr, m.sdf);   // K4: the result is not a reason to stop
    return m;
}

// PxMin(x, PxMin(y, z)) over a mesh scale, selectMin spelled (K3).
float ttkMinScale(float3 s)
{
    const float yz = s.y < s.z ? s.y : s.z;
    return s.x < yz ? s.x : yz;
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void triangleTriangleCollision(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint tid = gtid.x;
    const uint pairIdx = gid.x;                         // :1314, one block per pair
    if (pairIdx >= gTc.numWorkItems)                    // :1316-1317, block-uniform
        return;

    // ---- :1321-1356, per thread instead of staged --------------------------
    const PxgContactManagerInputGpu cin = Ptr<PxgContactManagerInputGpu>(gTc.cmInputs)[pairIdx];
    Ptr<PxgShapeGpu> shapes = Ptr<PxgShapeGpu>(gTc.shapes);
    const PxgShapeGpu e0 = shapes[cin.shapeRef0];
    const PxgShapeGpu e1 = shapes[cin.shapeRef1];
    Ptr<PxsCachedTransformGpu> xforms = Ptr<PxsCachedTransformGpu>(gTc.transformCache);
    const PxsCachedTransformGpu x0 = xforms[cin.transformCacheRef0];
    const PxsCachedTransformGpu x1 = xforms[cin.transformCacheRef1];
    const TtkMesh m0 = ttkReadMesh(e0.hullOrMeshPtr);
    const TtkMesh m1 = ttkReadMesh(e1.hullOrMeshPtr);

    Ptr<float> cd = Ptr<float>(gTc.contactDistances);
    const float contactDistance = cd[cin.transformCacheRef0] + cd[cin.transformCacheRef1];

    // ---- :1358-1360 (K2) -------------------------------------------------------
    const bool doFirst = (m0.sdf.texObject != 0u) &&
        ((m1.sdf.texObject == 0u)
         || ((m0.sdf.sdfDims.w == 0u) && (m1.sdf.sdfDims.w != 0u))
         || ((m1.sdf.sdfDims.w == m0.sdf.sdfDims.w) && (m1.numVerts < m0.numVerts)));

    // ---- :1368/:1375, the overlap flag: one byte of a word four pairs share,
    // written by triTriOverlapBda (ordered before this dispatch by the per-launch
    // barrier, P9.85).
    const uint ow = Ptr<uint>(gTc.overlaps + uint64_t(pairIdx >> 2u) * 4UL)[0];
    const bool generateContacts = ((ow >> ((pairIdx & 3u) * 8u)) & 0xFFu) != 0u;

    // ---- K1: the operands of doTriangleTriangleCollision, exchanged once -------
    TtkMesh mesh0, mesh1;
    PxgShapeGpu shape0, shape1;
    PxsCachedTransformGpu xf0, xf1;
    if (doFirst)
    {
        mesh0 = m1; mesh1 = m0; shape0 = e1; shape1 = e0; xf0 = x1; xf1 = x0;
    }
    else
    {
        mesh0 = m0; mesh1 = m1; shape0 = e0; shape1 = e1; xf0 = x0; xf1 = x1;
    }
    const bool isFlipped = doFirst;

    const float3 s0 = float3(shape0.scaleX, shape0.scaleY, shape0.scaleZ);
    const float4 r0 = float4(shape0.scaleRotX, shape0.scaleRotY, shape0.scaleRotZ, shape0.scaleRotW);
    const float3 s1 = float3(shape1.scaleX, shape1.scaleY, shape1.scaleZ);
    const float4 r1 = float4(shape1.scaleRotX, shape1.scaleRotY, shape1.scaleRotZ, shape1.scaleRotW);
    const float4 t0q = float4(xf0.qx, xf0.qy, xf0.qz, xf0.qw);
    const float3 t0p = float3(xf0.px, xf0.py, xf0.pz);
    const float4 t1q = float4(xf1.qx, xf1.qy, xf1.qz, xf1.qw);
    const float3 t1p = float3(xf1.px, xf1.py, xf1.pz);

    // ---- :1022, aToB = transform1.transformInv(transform0), in the engine's own
    // spelling (PxTransform.h:167-168): qinv.rotate(src.p - p), qinv * src.q.
    const float4 t1qInv = quatConjugate(t1q);
    const float4 aToBq = quatMul(t1qInv, t0q);
    const float3 aToBp = quatRotate(t1qInv, t0p - t1p);
    // K5: the pass-1 gather's materialised inverse (:1078).
    float3 bToAp;
    float4 bToAq;
    ttgGetInverse(aToBp, aToBq, bToAp, bToAq);

    // ---- :1016-1027 -------------------------------------------------------------
    if (tid == 0u)
        gTtfNumSubdivide = 0u;
    ttbInitAssignState(tid);
    uint numPatchesToKeep = 0u;
    GroupMemoryBarrierWithGroupSync();

    const bool mesh0HasSdf = mesh0.sdf.texObject != 0u;

    // ---- :1034-1221 -------------------------------------------------------------
    for (uint outer = 0u; outer < 2u && generateContacts; ++outer)
    {
        float cullScale = contactDistance;          // K3: every pass
        uint nbTriangles;
        SparseSdfTexture tex;
        if (outer == 0u)
        {
            nbTriangles = mesh0.numTris;
            cullScale /= ttkMinScale(s1);
            tex = sdfTexInitialize(mesh1.sdf);
        }
        else
        {
            if (mesh0HasSdf)                        // K4
            {
                nbTriangles = mesh1.numTris;
                cullScale /= ttkMinScale(s0);
                tex = sdfTexInitialize(mesh0.sdf);
            }
            else break;
        }
        // :1057 is not needed: the texture is per thread.

        // The gather's operands for this pass, selected once (:1070-1079).
        const bool pass0 = outer == 0u;
        const uint64_t gInd = pass0 ? mesh0.indices : mesh1.indices;
        const uint64_t gVerts = pass0 ? mesh0.verts : mesh1.verts;
        const float3 gS0 = pass0 ? s0 : s1;
        const float4 gR0 = pass0 ? r0 : r1;
        const float3 gS1 = pass0 ? s1 : s0;
        const float4 gR1 = pass0 ? r1 : r0;
        const float3 gP = pass0 ? aToBp : bToAp;    // K5
        const float4 gQ = pass0 ? aToBq : bToAq;

        uint numTrianglesToSubdividePerThread = 0u; // :1062
        uint i = 0u;

        // K8: the register copy.
        while (i < nbTriangles || numTrianglesToSubdividePerThread > 0u)
        {
            TtbContact c;                           // K6
            c.pos = float3(0.0f, 0.0f, 0.0f);
            c.separation = TTC_FLT_MAX;
            c.normal = float3(0.0f, 0.0f, 0.0f);
            c.candidate = false;

            const uint nbFoundTriangles = ttfFindInterestingTriangles(tid, nbTriangles, gInd, gVerts,
                                                                      gS0, gR0, gS1, gR1, cullScale, tex,
                                                                      gP, gQ, i, numTrianglesToSubdividePerThread);
            numTrianglesToSubdividePerThread = 0u;  // :1080

            const uint2 entry = ttfIndexGet(tid);                           // :1082-1083
            const uint ind = tid < nbFoundTriangles ? entry.x : nbTriangles;
            const uint subInd = tid < nbFoundTriangles ? entry.y : 0u;
            const uint safeInd = ind < nbTriangles ? ind : 0u;              // K9

            bool needsRefinement = false;
            if (ind < nbTriangles)
            {
                float3 dir = float3(0.0f, 0.0f, 0.0f);                      // K6
                float3 v0, v1, v2;
                if (pass0)
                {
                    ttcPass0Vertices(mesh0.indices, mesh0.verts, s0, r0, aToBp, aToBq, safeInd, subInd,
                                     s1, r1, v0, v1, v2);
                    const float sep = sdfDoTriangleCollision(tex, v0, v1, v2, c.pos, dir, cullScale);
                    ttcPass0Post(v0, v1, v2, sep, dir, tex.sdfBoxLower, tex.sdfBoxHigher, s1, r1,
                                 mesh0HasSdf, cullScale, contactDistance, TTK_REFINEMENT_RATIO_SQUARED,
                                 c.pos, c.separation, c.normal, c.candidate, needsRefinement);
                }
                else
                {
                    ttcPass1Vertices(mesh1.indices, mesh1.verts, s1, r1, aToBp, aToBq, safeInd,
                                     s0, r0, v0, v1, v2);
                    const float sep = sdfDoTriangleCollision(tex, v0, v1, v2, c.pos, dir, cullScale);
                    ttcPass1Post(sep, dir, s0, r0, aToBp, aToBq, cullScale, contactDistance,
                                 c.pos, c.separation, c.normal, c.candidate);
                }
            }

            // K7
            const uint newBufferCount = ttfAddToRefinementBuffer(tid, needsRefinement, ind, subInd, 0u,
                                                                 TTK_MAX_REFINEMENT_LEVEL);
            if (tid == 0u)
                gTtfNumSubdivide = newBufferCount;                           // :1199-1200

            const uint generatedThisIter = ttbBlockCountAndBroadcast(tid, c.candidate);   // :1202
            GroupMemoryBarrierWithGroupSync();                                  // :1204

            numTrianglesToSubdividePerThread = gTtfNumSubdivide;               // :1209

            ttbAssignContacts(tid, c, gTc.clusterBias, int(generatedThisIter), numPatchesToKeep);  // :1213-1220
        }
    }

    // ---- :1227-1230, the tail: every thread, block-uniformly, also when no
    // contacts were generated (it still writes the no-touch state).  Built member
    // by member from the push block (a whole-block load hides unread fields).
    TtbTailArgs a;
    a.cmOutputs = gTc.cmOutputs;
    a.materials = gTc.materials;
    a.contactStream = gTc.contactStream;
    a.patchStream = gTc.patchStream;
    a.counters = gTc.counters;
    a.touchChangeFlags = gTc.touchChangeFlags;
    a.patchChangeFlags = gTc.patchChangeFlags;
    a.startContactPatches = gTc.startContactPatches;
    a.startContactPoints = gTc.startContactPoints;
    a.startContactForces = gTc.startContactForces;
    a.patchBytesLimit = gTc.patchBytesLimit;
    a.contactBytesLimit = gTc.contactBytesLimit;
    a.forceBytesLimit = gTc.forceBytesLimit;
    // The LOCAL transform1 and the LOCAL shapes' materials: the tail undoes the
    // exchange itself (pxgSlangTriTriPatch note 4).
    ttbWritePatchesAndContacts(tid, pairIdx, numPatchesToKeep, t1q, t1p, isFlipped,
                               shape0.materialIndex, shape1.materialIndex, a);
}
