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

// fcVertexConvex.slang
//
// The Phase-1 battery for the cloth VERTEX vs convex contact path
// (femClothPrimitives.cu clothVertexConvexCollision + closestVertToConvex +
// computeClosestPolygon + distancePointSegment, plus
// deformableElementFilter.cuh find(PxgRigidFilterPair*)).
//
// P9.21 -- THE ALGEBRA MOVED TO pxgSlangVertexConvexCloth.  This shader keeps
// its entry point and its battery, which is what verifies the lift: the battery
// compares against a C++ host mirror of the CUDA, so it goes on measuring the
// core against an INDEPENDENT reference after the move, not against a second
// copy of itself.  What remains here is exactly what the engine kernel
// cloth_convexVertexContactGenLaunch also has to supply for itself:
//
//   * the LIVE attachment-filter check -- find(filterPairs, nbPairs, rigidId,
//     PxEncodeClothIndex(clothId, vertexIdx)), a binary search over the
//     PxgRigidFilterPair list navigated by the 3-key (index0, index1, index2)
//     lexicographic compare and accepting on exact match OR (ignoreIndex2)
//     index0/index1 equality.  A filtered vertex generates NOTHING, and the
//     quantifier is a SINGLE match -- the triangle sibling next door skips only
//     when all three corners match;
//   * the case: the world cloth vertex, the rigid pose, the convex scale and
//     scale rotation, the ids and the two distances.
//
// The hull is read through pxgSlangConvexHull's BDA accessors inside the
// module.  Its section offsets were checked against this file's former
// uint-offset accessors, which are now gone, byte for byte:
//     hullDesc        = hull + 16                              (both)
//     vertices        = hull + 48                              (both)
//     planes          = vertices + 16*nbVerts                  (both)
//     polyDescs       = planes   + 16*nbPolys                  (both)
//     verticesByEdges = polyDescs + 4*nbPolys                  (both)
//     facesByEdges    = verticesByEdges + 4*nbEdges            (both)
//     vertexData8     = facesByEdges + 2*nbEdges + 3*nbVerts   (both)
// and the field decodes agree too (nbPolygons = desc & 0xff, nbVerts =
// (desc >> 8) & 0xff, nbEdges = desc >> 16; polyDesc nbVerts = (d >> 8) & 0xff,
// vRef8 = d >> 16).

import pxgSlangCommon;
// The lifted core.  FvcCase and FvcWriter live there -- one definition, two
// users: this battery and the engine-ABI kernel.
import pxgSlangVertexConvexCloth;

struct FvcParams
{
    uint numCases; uint maxContacts; uint nbFilterPairs; uint pad0;   // @0
    uint64_t hullAddr;        // @16  base of the hull blob (FvcCase.ids2.y is a UINT offset into it)
    // PxgFEMContactWriter, by address (@24..@88)
    uint64_t wPoint, wNormalPen, wBary, wInfo, wCount, wByX, wTempByX, wIdxByX;
}

// PxgRigidFilterPair (32 bytes): u64 index0, u64 index1, u32 index2, pad
struct FilterPair
{
    uint2 index0;
    uint2 index1;
    uint index2;
    uint pad0, pad1, pad2;
}

[[vk::push_constant]]
ConstantBuffer<FvcParams> gParams;

RWStructuredBuffer<FvcCase> gCase;        // binding 0 (test input; not the engine ABI)
RWStructuredBuffer<FilterPair> gFilter;   // binding 1 (sorted)
RWStructuredBuffer<float4> gDebug;        // binding 2: (filtered, pen, wrote, insideBranch)

// ---- find(PxgRigidFilterPair*) (deformableElementFilter.cuh) ----------------
bool u64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
bool u64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

// 3-key lexicographic compare; accept on exact match OR (ignoreIndex2)
// index0/index1 equality
bool findFilter(uint2 rigidId, uint clothMask)
{
    const uint2 q1 = uint2(clothMask, 0);
    uint l = 0, r = gParams.nbFilterPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2;
        const FilterPair p = gFilter[mid];
        int cmp;
        if (!u64Eq(rigidId, p.index0))
            cmp = u64Lt(rigidId, p.index0) ? -1 : 1;
        else if (!u64Eq(q1, p.index1))
            cmp = u64Lt(q1, p.index1) ? -1 : 1;
        else if (0u != p.index2)
            cmp = (0u < p.index2) ? -1 : 1;
        else
            cmp = 0;
        if (cmp == 0 || (u64Eq(rigidId, p.index0) && u64Eq(q1, p.index1)))
            return true;
        if (cmp < 0)
            r = mid;
        else
            l = mid + 1;
    }
    return false;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void ctFcVertexConvex(uint3 tid : SV_DispatchThreadID)
{
    const uint caseIdx = tid.x;
    if (caseIdx >= gParams.numCases) return;
    const FvcCase cs = gCase[caseIdx];

    // LIVE attachment filter (a hit generates nothing)
    const uint2 rigidId = uint2(cs.ids.x, cs.ids.y);
    const uint clothMask = (cs.ids.z << FVC_BITSHIFT_ELEMENT_ID) | cs.ids.w;
    if (findFilter(rigidId, clothMask))
    {
        gDebug[caseIdx] = float4(1, 0, 0, 0);
        return;
    }

    // FvcCase.ids2.y stays a UINT offset into the blob; the address is formed here.
    const uint64_t hull = gParams.hullAddr + uint64_t(cs.ids2.y) * 4UL;

    FvcWriter w;
    w.point = gParams.wPoint; w.normalPen = gParams.wNormalPen; w.bary = gParams.wBary;
    w.info = gParams.wInfo;   w.count = gParams.wCount;
    w.byX = gParams.wByX;     w.tempByX = gParams.wTempByX; w.idxByX = gParams.wIdxByX;
    w.maxContacts = gParams.maxContacts;

    gDebug[caseIdx] = fcClothVertexConvexContactGen(cs, hull, w);
}
