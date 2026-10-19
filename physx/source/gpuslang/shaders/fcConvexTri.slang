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

// fcConvexTri.slang
//
// Slang port of the cloth convex-vs-triangle contact generation
// (femClothPrimitives.cu convexTriangleContactGen + selectPolygonFeatureIndex2,
// convexTriangle.cuh convexMeshSAT<false> / satConvexFaceNormals /
// satTriangleNormal / selectPolygonFeatureIndex, contactReduction.cuh
// contactReduce<false, true, 5>, PxgOutputIndex, and
// PxgRigidDbContactWriter::writeSurfaceContact (PhysX 5.12)):
//
// - convexScaledVerts precompute (vertex2Shape of every hull vertex);
// - convexMeshSAT<false>: satConvexFaceNormals (per convex face, plane
//   normal to shape space via vertex2ShapeNormalVector with the
//   1/normalize() invMagnitude scaling of the plane d, disjoint when either
//   projection separates beyond contactDist, running max of min1 - max0;
//   also fills the convexPlaneN/convexPlaneD caches) then satTriangleNormal
//   (hull projected onto the triangle normal, separation = min1 - triProj,
//   feature = TRI_FEATURE_IDX, normal = -n0).  The edge-edge stage is
//   compiled out for cloth (doEdgeEdge = false), so EDGE_FEATURE_IDX is
//   unreachable here; selectPolygonFeatureIndex still checks both.
// - selectPolygonFeatureIndex: for TRI/EDGE features pick the convex face
//   whose cached normal best aligns with minNormal, then scan the edges:
//   the best polyIndex2 neighbor of the winning face (max squared dot
//   among edges containing it), and if some edge normal-sum beats the face
//   alignment (dp > 0 and sqDp > maxDpSq * |edgeN|^2, seeded with
//   minProj^2), both of that edge's faces win instead (ordered by dot).
// - the two-face contact loop: reference plane from the vertex-space plane
//   via vertex2ShapeNormalVector + PxPlane::normalize (n /= |n|, d /= |n|),
//   polygon verts gathered through polyDesc vRef8/vertexData8 + vertex2Shape,
//   polyClip4 (tranche 95, inlined here), computeBarycentric VALIDITY gate
//   on each generated contact (the eps window is live in this caller),
//   contactReduce<clustering=false, keepAnotherDeepestForPCM=true,
//   maxPoints=5, dupeTest=true> when more than 5 survive, PxgOutputIndex
//   (ONE atomicAdd of popc(mask) per batch reserving a contiguous range —
//   the reservation happens BEFORE the `mask > 0` write gate, and that gate
//   is a SIGNED int comparison: a kept lane-31 contact makes the whole
//   batch unwritten while the counter still advances — faithful quirk),
//   then writeSurfaceContact (5.12): world-space point (w = restDist) and
//   normal (w = separation), rigidRef = rigidId.getInd() (u64), the PxU16
//   rigid material index, triRef = PxgEncodeDeformableTriangle(clothId,
//   triangleId), and the barycentric.
//
// WARP SERIALIZATION: the CUDA runs one warp per pair.  All reductions are
// order-fixed (warpReduction winnerLane = LOWEST lane holding the maximum,
// after per-lane strided scans where a strictly-greater candidate replaces),
// and the only atomic is the single PxgOutputIndex reservation per batch —
// so one CASE per thread with faithful two-level (per-lane stride, then
// lowest-lane tie) argmax loops is output-identical.  Caller glue not
// ported (documented): LoadShapePairWarp/transform caches, the filter-pair
// all-three-verts skip (find() ported in tranche 75), and the triangle
// transform into convex space (cases carry triLocVerts/triangleLocNormal).
//
// Hull blob layout (ConvexScratch getters): centerOfMass float4, header
// uint4 (x = nbEdgesNbHullVerticesNbPolygons: bits 0-7 nbPolygons, 8-15
// nbVerts, 16-31 nbEdges), extents float4, then vertices float4[nbVerts],
// planes float4[nbPolygons], polyDescs u32[nbPolygons] (bits 0-7 minIndex,
// 8-15 nbVerts, 16-31 vRef8), verticesByEdges u16[2*nbEdges], facesByEdges
// u8[2*nbEdges], facesByVertices u8[3*nbVerts], vertexData8 u8[].

import pxgSlangCommon;
// P9.15c: the hull is read through the SHARED BDA accessors instead of a bound
// StructuredBuffer.  pxgSlangConvexHull already decodes this exact blob for the
// particle-vs-hull path (P9.14b), and its section offsets were verified to match
// this file's former uint-offset accessors byte for byte:
//     verticesByEdges = hull + 48 + 16*nbVerts + 20*nbPolys   (both)
//     facesByEdges    = verticesByEdges + 4*nbEdges           (both)
//     vertexData8     = facesByEdges + 2*nbEdges + 3*nbVerts  (both)
// Moving to them removes the second spelling AND puts this body one step from
// being callable by the engine-ABI kernel, which cannot bind a StructuredBuffer.
import pxgSlangConvexHull;

import pxgSlangCommon;
// P9.15c: the algebra moved to a module so the engine-ABI kernel can call it.
// This shader keeps its entry point and its battery, which is what verifies the
// lift -- see pxgSlangConvexTriCloth.slang.
import pxgSlangConvexTriCloth;
import pxgSlangDeformableWriters;   // PxgRigidDbContactWriterGpu

struct FctParams
{
    uint numCases; uint maxContacts; uint pad0; uint pad1;   // @0
    uint64_t hullAddr;        // @16  base of the hull blob (FctCase.ids2.y is a UINT offset into it)
    // PxgRigidDbContactWriter, by address (@24..@96)
    uint64_t wPoint, wNormalPen, wBary, wInfo, wCount, wByX, wTempByX, wIdxByX;
    uint64_t wRigidMat;       // @88  PxU16* outRigidMaterialIndex (5.12)
}

// struct FctCase now lives in pxgSlangConvexTriCloth (public) -- one definition,
// two users: this battery and the engine-ABI kernel that will fill one.

[[vk::push_constant]]
ConstantBuffer<FctParams> gParams;

RWStructuredBuffer<FctCase> gCase;     // binding 0 (test input; not part of the engine ABI)
RWStructuredBuffer<uint4> gDebug;      // binding 1: per case (isSATPassed, featureIndex, face1, face2)

[shader("compute")]
[numthreads(64, 1, 1)]
void ctFcConvexTri(uint3 tid : SV_DispatchThreadID)
{
    const uint caseIdx = tid.x;
    if (caseIdx >= gParams.numCases) return;
    const FctCase cs = gCase[caseIdx];
    // FctCase.ids2.y stays a UINT offset into the blob; the address is formed here.
    const uint64_t hull = gParams.hullAddr + uint64_t(cs.ids2.y) * 4UL;

    PxgRigidDbContactWriterGpu w;
    w.outPoint = gParams.wPoint; w.outNormalPen = gParams.wNormalPen; w.outBarycentric = gParams.wBary;
    w.outContactInfo = gParams.wInfo; w.outRigidMaterialIndex = gParams.wRigidMat;
    w.totalContactCount = gParams.wCount;
    w.contactByRigid = gParams.wByX; w.tempContactByRigid = gParams.wTempByX;
    w.contactIndexSortedByRigid = gParams.wIdxByX;
    w.maxNumContacts = gParams.maxContacts;

    gDebug[caseIdx] = fcClothConvexTriContactGen(cs, hull, w);
}
